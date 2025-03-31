import time
from typing import Tuple

import numpy as np
import scipy.integrate

from . import abstract_base_classes, utils


class GonzalezDiscreteGradient(abstract_base_classes.DiscreteGradient):
    def compute(
        self,
        system_n,
        system_n1,
        system_n05,
        func_name: str,
        jacobian_name: str,
        argument_n: np.ndarray,
        argument_n1: np.ndarray,
        increment_tolerance: float = 1e-12,
        **kwargs,
    ) -> np.ndarray:
        func_n = getattr(system_n, func_name)()
        func_n1 = getattr(system_n1, func_name)()
        midpoint_jacobian = getattr(system_n05, jacobian_name)()
        midpoint_jacobian, func_n, func_n1 = adjust_midpoint_jacobian(
            midpoint_jacobian, func_n, func_n1
        )
        return Gonzalez_discrete_gradient(
            func_n,
            func_n1,
            midpoint_jacobian,
            argument_n,
            argument_n1,
            increment_tolerance,
        )


class GonzalezDecomposedDiscreteGradient(abstract_base_classes.DiscreteGradient):
    def compute(
        self,
        system_n,
        system_n1,
        system_n05,
        func_name: str,
        jacobian_name: str,
        argument_n: np.ndarray,
        argument_n1: np.ndarray,
        nbr_func_parts: int,
        func_parts_n,
        func_parts_n1,
        increment_tolerance: float = 1e-12,
        **kwargs,
    ) -> np.ndarray:
        discrete_gradient = []

        for index in range(nbr_func_parts):
            func_n = getattr(system_n, f"{func_name}_{index+1}")()
            func_n1 = getattr(system_n1, f"{func_name}_{index+1}")()
            midpoint_jacobian = getattr(system_n05, f"{jacobian_name}_{index+1}")()
            midpoint_jacobian, func_n, func_n1 = adjust_midpoint_jacobian(
                midpoint_jacobian, func_n, func_n1
            )

            part_argument_n = system_n.decompose_state()[func_parts_n[index]]
            part_argument_n1 = system_n1.decompose_state()[func_parts_n1[index]]

            contribution = Gonzalez_discrete_gradient(
                func_n,
                func_n1,
                midpoint_jacobian,
                part_argument_n,
                part_argument_n1,
                increment_tolerance,
            )

            discrete_gradient.append(contribution.squeeze())

        return np.concatenate(discrete_gradient, axis=0)


class MeanValueDiscreteGradient(abstract_base_classes.DiscreteGradient):
    def compute(
        self,
        system_n,
        system_n1,
        system_n05,
        func_name: str,
        jacobian_name: str,
        argument_n: np.ndarray,
        argument_n1: np.ndarray,
        increment_tolerance: float = 1e-12,
        **kwargs,
    ) -> np.ndarray:
        # define gradient-function
        evaluate = lambda interpolation_value: getattr(
            utils.interpolate_system(
                system=system_n,
                x0=argument_n,
                x1=argument_n1,
                interpolation_value=interpolation_value,
            ),
            jacobian_name,
        )()
        integration_method = kwargs.pop("integration_method", "5-point")
        match integration_method:
            case "scipy":
                accuracy = kwargs.pop("scipy_accuracy", 1e-12)
                discrete_gradient = np.array(
                    [
                        scipy.integrate.quad(
                            func=get_ith_entry,
                            args=(evaluate, i),
                            a=0,
                            b=1,
                            epsabs=accuracy,
                            epsrel=accuracy,
                        )[0]
                        for i in range(argument_n.shape[0])
                    ]
                )
            case _:
                quad_order = 5
                discrete_gradient = gauss_integrate_function(
                    func=evaluate, quad_order=quad_order
                )

        return discrete_gradient.squeeze()


class CoordIncDiscreteGradient(abstract_base_classes.DiscreteGradient):
    def compute(
        self,
        system_n,
        system_n1,
        system_n05,
        func_name: str,
        jacobian_name: str,
        argument_n: np.ndarray,
        argument_n1: np.ndarray,
        increment_tolerance: float = 1e-12,
        **kwargs,
    ) -> np.ndarray:
        state_n = system_n.state
        state_n1 = system_n1.state
        evaluate_func = lambda state: getattr(system_n.copy(state=state), func_name)()
        # initialize with midpoint jacobian
        midpoint_jacobian = getattr(system_n05, jacobian_name)()
        discrete_gradient = midpoint_jacobian
        # compute coord-inc-gradient
        for i in range(discrete_gradient.shape[0]):
            arg1, arg2 = arguments_coord_inc(index=i, x0=state_n, x1=state_n1)
            current_variable_initial, current_variable_inc = (
                state_n[i],
                state_n1[i],
            )
            if (
                abs(current_variable_inc - current_variable_initial)
                > increment_tolerance
            ):
                discrete_gradient[i] = (evaluate_func(arg1) - evaluate_func(arg2)) / (
                    current_variable_inc - current_variable_initial
                )
        return discrete_gradient.squeeze()


class DiscreteGradientFactory:
    """Factory for creating discrete gradient instances."""

    @staticmethod
    def create(type: str) -> abstract_base_classes.DiscreteGradient:
        if type == "Gonzalez":
            return GonzalezDiscreteGradient()
        elif type == "Gonzalez_decomposed":
            return GonzalezDecomposedDiscreteGradient()
        elif type == "MeanValue":
            return MeanValueDiscreteGradient()
        elif type == "CoordinateIncrement":
            return CoordIncDiscreteGradient()
        else:
            raise ValueError(f"Unsupported discrete gradient type: {type}")


def discrete_gradient(
    system_n,
    system_n1,
    system_n05,
    func_name: str,
    jacobian_name: str,
    argument_n: np.ndarray,
    argument_n1: np.ndarray,
    type: str = "Gonzalez",
    increment_tolerance: float = 1e-12,
    **kwargs,
):
    gradient_computer = DiscreteGradientFactory.create(type)

    return gradient_computer.compute(
        system_n=system_n,
        system_n1=system_n1,
        system_n05=system_n05,
        func_name=func_name,
        jacobian_name=jacobian_name,
        argument_n=argument_n,
        argument_n1=argument_n1,
        increment_tolerance=increment_tolerance,
        **kwargs,
    )


def Gonzalez_discrete_gradient(
    func_n,
    func_n1,
    midpoint_jacobian,
    argument_n,
    argument_n1,
    denominator_tolerance,
):
    """Compute the discrete gradient using the Gonzalez approach."""
    discrete_gradient = midpoint_jacobian
    increment = argument_n1 - argument_n
    denominator = increment.T @ increment
    # print(func_n1[0] - func_n[0])
    if denominator > denominator_tolerance:
        for index in range(midpoint_jacobian.shape[0]):
            discrete_gradient[index, :] += (
                (
                    func_n1[index]
                    - func_n[index]
                    - np.dot(midpoint_jacobian[index, :], increment)
                )
                / denominator
                * increment.T
            )
            # print(discrete_gradient[index, :])
        result = discrete_gradient

    else:
        result = midpoint_jacobian
    return result.squeeze()


def adjust_midpoint_jacobian(midpoint_jacobian, func_n, func_n1):
    """Helper function to adjust the midpoint Jacobian and function evaluations for scalar-valued functions."""
    if midpoint_jacobian.ndim == 1:
        return (
            midpoint_jacobian[np.newaxis, :],
            np.array([func_n]),
            np.array([func_n1]),
        )
    return midpoint_jacobian, func_n, func_n1


def interpolate_vectors(loc: float, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Helper function to linearly interpolate between vectors vec1 and vec2"""
    return np.multiply(1 - loc, vec1) + np.multiply(loc, vec2)


def gauss_integrate_function(func, quad_order) -> np.ndarray:
    """Helper function to integrate func(x) from x=0 to x=1 using gauss integration"""
    x_i, weights = gauss_constants(quad_order)
    contributions = [func(0.5 * x + 0.5) * weight for x, weight in zip(x_i, weights)]
    return 0.5 * np.add.reduce(contributions)


def gauss_constants(quad_order):
    """returns (x_i, weights) as a tuple"""
    return {
        2: (np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]), np.array([1, 1])),
        3: (
            np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]),
            np.array([5 / 9, 8 / 9, 5 / 9]),
        ),
        5: (
            np.array(
                [
                    -np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
                    -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
                    0,
                    np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
                    np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
                ]
            ),
            np.array(
                [
                    (322 - 13 * np.sqrt(70)) / (900),
                    (322 + 13 * np.sqrt(70)) / (900),
                    128 / 225,
                    (322 + 13 * np.sqrt(70)) / (900),
                    (322 - 13 * np.sqrt(70)) / (900),
                ]
            ),
        ),
    }.pop(quad_order)


def arguments_coord_inc(index, x0, x1) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.array([*x1[: index + 1], *x0[index + 1 :]]),
        np.array([*x1[:index], *x0[index:]]),
    )


def get_ith_entry(arg, vector_function, i):
    """return i-th entry of a single-parameter vector function"""
    return vector_function(arg)[i]

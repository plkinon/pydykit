import plotly.graph_objects as go

import pydykit
import pydykit.postprocessors as postprocessors
import pydykit.plotters as plotters

manager = pydykit.managers.Manager()

name = "spring_pendulum"
path_config_file = f"./pydykit/example_files/{name}.yml"

manager.configure_from_path(path=path_config_file)

result = pydykit.results.Result(manager=manager)
result = manager.manage(result=result)
df = result.to_df()

fig = go.Figure(
    data=go.Scatter3d(
        x=df["position0_particle0"],
        y=df["position1_particle0"],
        z=df["position2_particle0"],
        marker=dict(
            size=3,
            color=df["time"],
            colorscale="Viridis",
            colorbar=dict(
                thickness=20,
                title="time",
            ),
        ),
        line=dict(
            color="darkblue",
            width=3,
        ),
    )
)

fig.show()

postprocessor = postprocessors.Postprocessor(
    manager,
    state_results_df=df,
)

plotter = plotters.Plotter(results_df=postprocessor.results_df)


postprocessor.postprocess(
    quantities_and_evaluation_points={
        "total_energy": ["current_time", "interval_increment"]
    }
)

fig01 = plotter.visualize_time_evolution(quantities=["total_energy_current_time"])
fig01.show()

fig02 = plotter.visualize_time_evolution(quantities=["total_energy_interval_increment"])
fig02.show()

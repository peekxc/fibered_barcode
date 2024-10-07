from bokeh.plotting import figure, curdoc
from bokeh.layouts import column
from bokeh.models import Range1d, Slider, ColumnDataSource, PointDrawTool, TapTool, Div
import numpy as np


# Initial slope and intercept
p1 = np.array([0.2, 0.2])
p2 = np.array([0.8, 0.8])
m_init = (p2[1] - p1[1]) / (p2[0] - p1[0])
b_init = p1[1] - m_init * p1[0]

# Generate initial data for the line
x_vals = np.linspace(-10, 10, 400)
y_vals = m_init * x_vals + b_init
points_source = ColumnDataSource(data=dict(x=[p1[0], p2[0]], y=[p1[1], p2[1]]))
line_source = ColumnDataSource(data=dict(x=x_vals, y=y_vals))

p = figure(title="Line Selection Window", x_axis_label="x", y_axis_label="y")
p.x_range = Range1d(0, 1)
p.y_range = Range1d(0, 1)
lr = p.line("x", "y", source=line_source, line_width=2)
sr = p.scatter("x", "y", source=points_source, size=16, color="red", fill_alpha=0.20)

draw_tool = PointDrawTool(renderers=[sr], add=False, drag=True)
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool

line_text = Div(text=f"Slope: {m_init:.4f}, Intercept: {b_init:.4f}")


def update_line(x1, x2, y1, y2):
	"""Given two points (x1,y1) and (x2,y2), updates the line source data to show the line (y = mx + b)."""
	if x1 != x2:  # Avoid division by zero
		m = (y2 - y1) / (x2 - x1)
		b = y1 - m * x1
	else:
		m = np.inf
		b = 0
	x_vals = np.linspace(-10, 10, 400)
	y_vals = m * x_vals + b if m != np.inf else np.full_like(x_vals, np.nan)
	line_source.data = dict(x=x_vals, y=y_vals)
	line_text.text = f"Slope: {m:.4f}, Intercept: {b:.4f}"


def sync_line_data(attr, old, new):
	print("Changing line")
	x1, y1 = points_source.data["x"][0], points_source.data["y"][0]
	x2, y2 = points_source.data["x"][1], points_source.data["y"][1]
	update_line(x1, x2, y1, y2)


def on_mouse_move(event):
	selected_index = points_source.selected.indices
	# print(f"selected: {selected_index}, event: {event}")
	if len(selected_index) > 0:
		x1, y1 = (event.x, event.y) if selected_index[0] == 0 else (points_source.data["x"][0], points_source.data["y"][0])
		x2, y2 = (event.x, event.y) if selected_index[0] == 1 else (points_source.data["x"][1], points_source.data["y"][1])
		update_line(x1, x2, y1, y2)


points_source.on_change("data", sync_line_data)
p.on_event("mousemove", on_mouse_move)
layout = column(p, line_text)
curdoc().add_root(layout)


# Create Sliders for slope and intercept
# slope_slider = Slider(start=-5, end=5, value=m_init, step=0.1, title="Slope (m)")
# intercept_slider = Slider(start=-10, end=10, value=b_init, step=0.1, title="Intercept (b)")


# # Attach the callback to the sliders
# def update_slope_intercept(attr, old, new):
# 	m = slope_slider.value
# 	b = intercept_slider.value
# 	new_y = m * x_vals + b
# 	line_source.data = dict(x=x_vals, y=new_y)


# slope_slider.on_change("value", update_line)
# intercept_slider.on_change("value", update_line)

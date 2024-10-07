from itertools import *
from typing import Any, Callable, Collection, Sequence

import numpy as np
from bokeh.models import Span
from bokeh.plotting import figure, show
from numpy.typing import ArrayLike


## Meta-programming to the rescue!
def valid_parameters(el: Any, prefix: str = "", exclude: list = [], **kwargs):
	"""Extracts valid parameters for a bokeh plotting element.

	This function takes as input a bokeh model (i.e. figure, Scatter, MultiLine, Patch, etc.) and a set of keyword arguments prefixed by 'prefix',
	and extracts from the the given keywords a dictionary of valid parameters for the given element, with the chosen prefix removed.

	Example:

	  >>> valid_parameters(MultiLine, prefix="multiline_", multiline_linewidth=2.0)
	  # { 'linewidth' : 2.0 }

	Parameters:
	  el: bokeh model or glyph element with a .parameters() function
	  prefix: prefix to extract. Defaults to empty string.
	  kwargs: keyword arguments to extract the valid parameters from.

	"""
	assert hasattr(el, "parameters"), f"Invalid bokeh element '{type(el)}'; must have valid parameters() call"
	param_names = {param[0].name for param in el.parameters()}
	stripped_params = {k[len(prefix) :] for k in kwargs.keys() if k.startswith(prefix)}
	valid_params = {p: kwargs[prefix + p] for p in (stripped_params & param_names) if p not in exclude}
	return valid_params


def as_dgm(dgm: np.ndarray) -> np.ndarray:
	if (
		isinstance(dgm, np.ndarray)
		and dgm.dtype.names is not None
		and "birth" in dgm.dtype.names
		and "death" in dgm.dtype.names
	):
		return dgm
	dgm = np.atleast_2d(dgm)
	dgm_dtype = [("birth", "f4"), ("death", "f4")]
	return np.array([tuple(pair) for pair in dgm], dtype=dgm_dtype)


def figure_dgm(dgm: Sequence[tuple] = None, pt_size: int = 5, show_filter: bool = False, **kwargs):
	default_figkwargs = dict(width=300, height=300, match_aspect=True, aspect_scale=1, title="Persistence diagram")
	fig_kwargs = default_figkwargs.copy()
	if dgm is None or len(dgm) == 0:
		fig_kwargs["x_range"] = (0, 1)
		fig_kwargs["y_range"] = (0, 1)
		min_val = 0
		max_val = 1
	else:
		dgm = as_dgm(dgm)
		max_val = max(np.ravel(dgm["death"]), key=lambda v: v if v != np.inf else -v)
		max_val = max_val if max_val != np.inf else max(dgm["birth"]) * 5
		min_val = min(np.ravel(dgm["birth"]))
		min_val = min_val if min_val != max_val else 0.0
		delta = abs(min_val - max_val)
		min_val, max_val = min_val - delta * 0.10, max_val + delta * 0.10
		fig_kwargs["x_range"] = (min_val, max_val)
		fig_kwargs["y_range"] = (min_val, max_val)

	## Parameterize the figure
	from bokeh.models import PolyAnnotation

	fig_kwargs = valid_parameters(figure, **(fig_kwargs | kwargs))
	p = kwargs.get("figure", figure(**fig_kwargs))
	p.xaxis.axis_label = "Birth"
	p.yaxis.axis_label = "Death"
	polygon = PolyAnnotation(
		fill_color="gray",
		fill_alpha=1.0,
		xs=[min_val - 100, max_val + 100, max_val + 100],
		ys=[min_val - 100, min_val - 100, max_val + 100],
		line_width=0,
	)
	p.add_layout(polygon)
	# p.patch([min_val-100, max_val+100, max_val+100], [min_val-100, min_val-100, max_val+100], line_width=0, fill_color="gray", fill_alpha=0.80)

	## Plot non-essential points, where applicable
	if dgm is not None and any(dgm["death"] != np.inf):
		x = dgm["birth"][dgm["death"] != np.inf]
		y = dgm["death"][dgm["death"] != np.inf]
		p.scatter(x, y, size=pt_size)

	## Plot essential points, where applicable
	if dgm is not None and any(dgm["death"] == np.inf):
		x = dgm["birth"][dgm["death"] == np.inf]
		y = np.repeat(max_val - delta * 0.05, sum(dgm["death"] == np.inf))
		s = Span(dimension="width", location=max_val - delta * 0.05, line_width=1.0, line_color="gray", line_dash="dotted")
		s.level = "underlay"
		p.add_layout(s)
		p.scatter(x, y, size=pt_size, color="red")

	return p

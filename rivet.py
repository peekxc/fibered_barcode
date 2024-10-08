from itertools import combinations
import os
import pathlib
import subprocess
from tempfile import NamedTemporaryFile
from typing import Iterable, Union, Callable
import numpy as np
import splex as sx
import re
from shapely import Point
from shapely import GeometryCollection


## Meta-programming to the rescue!
def valid_parameters(el, prefix: str = "", exclude: list = [], **kwargs):
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
	valid_params = {p: kwargs[prefix + p] for p in (stripped_params & param_names) if not p in exclude}
	return valid_params


def bigraded_betti(
	S: sx.ComplexLike,
	f1: Callable,
	f2: Callable,
	p: int = 0,
	xbin: int = 15,
	ybin: int = 15,
	rivet_path: str = "~/rivet",
	verbose: int = 0,
	input_file: str = None,
	output_file: str = None,
	**kwargs,
):
	## Find RIVET CLI first
	rivet_path = os.path.expanduser(rivet_path)
	rivet_path = pathlib.Path(os.path.join(rivet_path, "rivet_console")).resolve()
	assert rivet_path.exists(), "Did not find file 'rivet_console'. Did you supply a valid 'rivet_path'?"
	rivet_path_str = str(rivet_path.absolute())

	## Prep the temporary file to hand off to rivet CLI
	tf = NamedTemporaryFile() if input_file is None else open(input_file, "w", encoding="utf-8")
	inp_file_str = "--datatype bifiltration\n"
	inp_file_str += "--xlabel density\n"
	inp_file_str += "--ylabel diameter\n"
	inp_file_str += "\n"
	# for s, fs_1, fs_2 in zip(S, f1(S), f2(S)):
	# 	inp_file_str += f"{' '.join([str(v) for v in s])} ; {fs_1} {fs_2}\n"
	## TODO: replace this
	from simplextree import SimplexTree

	rp = re.compile(r"[\[\(,\)\]]")
	if isinstance(S, SimplexTree):
		simplex_s = np.array([rp.sub("", str(s)).strip() for s in S.vertices])
		simplex_f = np.array([f" ; {f1([s])} {f2([s])}" for s in S.vertices])
		inp_file_str += "\n".join(np.char.add(simplex_s, simplex_f))
		inp_file_str += "\n"
		simplex_s = np.array([rp.sub("", str(s)).strip() + " ; " for s in S.edges])
		simplex_s = np.char.add(simplex_s, f1(S.edges).astype(str))
		simplex_s = np.char.add(simplex_s, np.repeat(" ", sx.card(S, 1)))
		simplex_s = np.char.add(simplex_s, f2(S.edges).astype(str))
		inp_file_str += "\n".join(simplex_s)
		inp_file_str += "\n"
		simplex_s = np.array([rp.sub("", str(s)).strip() + " ; " for s in S.triangles])
		simplex_s = np.char.add(simplex_s, f1(S.triangles).astype(str))
		simplex_s = np.char.add(simplex_s, np.repeat(" ", sx.card(S, 2)))
		simplex_s = np.char.add(simplex_s, f2(S.triangles).astype(str))
		inp_file_str += "\n".join(simplex_s)
	else:
		simplex_s = np.array([rp.sub("", str(s)).strip() for s in S])
		simplex_f = np.array([f" ; {f1(s)} {f2(s)}" for s in S])
		inp_file_str += "\n".join(np.char.add(simplex_s, simplex_f))

	with open(tf.name, "w") as inp_file:
		inp_file.write(inp_file_str)

	## Prep the rivet_console command
	output_tf = NamedTemporaryFile() if output_file is None else open(output_file, "w", encoding="utf-8")
	xreverse, yreverse = kwargs.get("xreverse", False), kwargs.get("yreverse", False)
	rivet_cmd = [rivet_path_str, inp_file.name]
	rivet_cmd += ["--betti", "--homology", str(p), "--xbins", str(xbin), "--ybins", str(ybin)]
	rivet_cmd += ["--xreverse" if xreverse else ""]
	rivet_cmd += ["--yreverse" if yreverse else ""]
	rivet_cmd += [">", output_tf.name]

	## Call rivet_console
	if verbose > 0:
		print(f"Calling rivet CLI with command: '{' '.join(rivet_cmd)}'")
	subprocess.run(" ".join(rivet_cmd), shell=True, check=True)
	# assert res == 0, "RIVET subprocess did not finish status code {res} "

	## Process the output file + return
	with open(output_tf.name) as f:
		rout = [line.rstrip("\n") for line in f]
	assert len(rout) > 0, "RIVET subprocess did not write any output."
	xi, yi, di, bi = [rout.index(key) for key in ["x-grades", "y-grades", "Dimensions > 0:", "Betti numbers:"]]
	b0, b1, b2 = [rout.index(key) for key in ["xi_0:", "xi_1:", "xi_2:"]]
	xg = np.array([eval(x) for x in rout[(xi + 1) : yi] if len(x) > 0])
	yg = np.array([eval(x) for x in rout[(yi + 1) : di] if len(x) > 0])

	def gen_tuples(L: Iterable, value: bool = True):
		for x in filter(lambda x: len(x) > 0, L):
			i, j, val = tuple(eval(x))
			yield (xg[i], yg[j], val) if value else (i, j, val)

	b_type = [("x", "f4"), ("y", "f4"), ("value", "i4")]
	betti_info = {}
	betti_info["x-grades"] = xg
	betti_info["y-grades"] = yg
	betti_info["hilbert"] = np.fromiter(gen_tuples(rout[(di + 1) : bi]), dtype=b_type)
	betti_info["0"] = np.fromiter(gen_tuples(rout[(b0 + 1) : b1]), dtype=b_type)
	betti_info["1"] = np.fromiter(gen_tuples(rout[(b1 + 1) : b2]), dtype=b_type)
	betti_info["2"] = np.fromiter(gen_tuples(rout[(b2 + 1) :]), dtype=b_type)
	return betti_info


def anchors(S: Union[np.ndarray, dict]):
	"""Computes the anchors of the set of bigraded Betti numbers S."""
	if isinstance(S, dict):
		b0 = np.c_[S["0"]["x"], S["0"]["y"]]
		b1 = np.c_[S["1"]["x"], S["1"]["y"]]
		S = np.vstack((b0, b1))
	assert isinstance(S, np.ndarray), "'S' must be the set of Betti numbers."
	anchors = set()
	for p, q in combinations(S, 2):
		if not (np.all(p <= q)) and (np.any(p < q) or p[0] == q[0] or p[1] == q[1]):
			lub = np.array([p, q]).max(axis=0)  # lowest upper bound
			anchors.add(tuple(lub))
	anchors = np.array(list(anchors), dtype=float) if len(anchors) > 0 else np.empty((0, 2), dtype=float)
	anchors = np.unique(np.vstack([anchors, S]), axis=0)
	return anchors


# def push_map(X: np.ndarray, a: float, b: float) -> np.ndarray:
# 	"""Projects set of points X in the plane onto the line given by f(x) = ax + b

# 	Returns:
# 		(x,y,f) = projected points (x,y) and the distance to the lowest projected point on the line.
# 	"""
# 	n = X.shape[0]
# 	out = np.zeros((n, 3))
# 	for i, (x, y) in enumerate(X):
# 		tmp = ((y - b) / a, y) if (x * a + b) < y else (x, a * x + b)
# 		dist_to_end = np.sqrt(tmp[0] ** 2 + (b - tmp[1]) ** 2)
# 		out[i, :] = np.append(tmp, dist_to_end)
# 	return out


## Faster form of the above
def push_map(X: np.ndarray, a: float, b: float):
	"""Projects 2-d points  to the line given in point-intercept form.

	Parameters:
		X = 2-d points
		a = slope
		b = intercept
	Returns:
		(x,y,f) = points (x,y) projected onto the line y = a * x + b along with the distance to the lowest projected point on the line.
	"""
	assert a >= 0, "Slope must be non-negative"
	X = np.atleast_2d(X)
	assert X.ndim == 2 and X.shape[1] == 2, "Only projects 2-dimensional points"
	x, y = X[:, 0], X[:, 1]
	cond = (x * a + b) < y
	tmp_0 = np.where(cond, (y - b) / a, x)
	tmp_1 = np.where(cond, y, a * x + b)
	return np.sqrt(tmp_0**2 + (b - tmp_1) ** 2)


def arrangement(anchors: np.ndarray, xg, yg):
	from shapely import LineString, MultiLineString, polygonize, get_parts, polygonize_full, Polygon, segmentize
	from shapely import LineString, multipolygons, segmentize, boundary, make_valid, envelope, union_all

	assert all(anchors[:, 0] > 0), "Not all lines have positive slope"
	C, D = anchors[:, 0], anchors[:, 1]
	bbox = [[np.min(xg), np.max(xg)], [np.min(yg), np.max(yg)]]
	ys, ye = np.array(bbox[1])
	xs, xe = np.array(bbox[0])
	YS = C * xs - D
	YE = C * xe - D
	bx_s, bx_e = xs, xe
	by_s, by_e = np.min(YS), np.max(YE)
	BB = LineString([(bx_s, by_s), (bx_e, by_s), (bx_e, by_e), (bx_s, by_e), (bx_s, by_s)])  ## bounding box
	LL = [LineString([[xs, ys], [xe, ye]]) for ys, ye in zip(YS, YE)]  ## lines intersecting the box
	A = union_all([*LL, BB])
	subdivision = polygonize(get_parts(A))
	template_points = np.array(
		[np.ravel(two_cell.centroid.xy) for two_cell in get_parts(subdivision) if two_cell.area > 1e-10]
	)
	return subdivision, template_points


def figure_betti(betti: dict, show_coords: bool = False, highlight: int = None, **kwargs):
	"""Creates a figure of the dimension function + the bigraded Betti numbers."""
	from bokeh.plotting import figure
	from map2color import rgb2hex
	from bokeh.palettes import linear_palette, gray

	BI = betti
	h = BI["hilbert"]
	xg, yg = BI["x-grades"], BI["y-grades"]
	x_step, y_step = np.abs(np.diff(xg)[0]), np.abs(np.diff(yg)[0])

	unique_vals = np.sort(np.unique(h["value"]))
	pal_offset = int(len(unique_vals) * 0.20)
	gray_pal = tuple(reversed(gray(255)))
	gray_pal = np.array(linear_palette(gray_pal[25:], pal_offset + len(unique_vals)))
	color_indices = np.searchsorted(unique_vals, h["value"])
	# box_intensity = ((1.0 - (h["value"] / max(h["value"]))) * 255).astype(int)
	# box_color = np.repeat(box_intensity, 3).reshape(len(h["value"]), 3).astype(np.uint8)

	from bokeh.models import ColumnDataSource

	highlight = -1 if highlight is None else int(highlight)
	fill_color = np.where(h["value"] == highlight, "blue", gray_pal[color_indices + pal_offset])
	r_source = ColumnDataSource(
		data=dict(x=h["x"] + x_step / 2.0, y=h["y"] + y_step / 2.0, fill_color=fill_color, value=h["value"])
	)
	r_tooltips = [("dim", "@value")]
	if show_coords:
		r_tooltips += [("coords", "(@x, @y)")]

	fig_kwargs = dict(width=250, height=250) | valid_parameters(figure, **kwargs)
	p = figure(**fig_kwargs, tooltips=r_tooltips)

	## Plot the rectangles
	from bokeh.models import Rect

	default_rect_params = dict(fill_alpha=0.50, line_color="black", line_alpha=0.0, line_width=0.0)
	rect_params = valid_parameters(
		Rect, exclude=["x", "y", "fill_color", "width", "height"], **(default_rect_params | kwargs)
	)
	rect_r = p.rect(x="x", y="y", fill_color="fill_color", width=x_step, height=y_step, source=r_source, **rect_params)
	p.hover.renderers = [rect_r]  # hover only for rectangles

	from bokeh.models import Scatter

	pt_num = (len(BI["0"]), len(BI["1"]), len(BI["2"]))
	pt_colors = [rgb2hex(c + [int(0.75 * 255)]) for c in [[52, 209, 93], [212, 91, 97], [254, 255, 140]]]
	scatter_params = valid_parameters(Scatter, exclude=["x", "y"], **(dict(size=8) | kwargs))
	print(scatter_params)
	p.scatter(BI["0"]["x"], BI["0"]["y"], **(scatter_params | dict(color=np.repeat(pt_colors[0], pt_num[0]))))
	p.scatter(BI["1"]["x"], BI["1"]["y"], **(scatter_params | dict(color=np.repeat(pt_colors[1], pt_num[1]))))
	p.scatter(BI["2"]["x"], BI["2"]["y"], **(scatter_params | dict(color=np.repeat(pt_colors[2], pt_num[2]))))
	return p


# def line_to_point(a: float, b: float):
# 	"""Converts a line of the form y = ax + b to a point (a,-b)."""
# 	return (a, -b)


# def point_to_line(c: float, d: float):
# 	"""Converts a point of the form (c,d) \in R^2 to a line y = cx - d."""
# 	return (a, -b)


class RIVET:
	def __init__(self, S, f: Callable, g: Callable) -> None:
		## Store the complex + filter functions f and g
		self.S = S
		self.f = f
		self.g = g
		self._f_cross_g = np.c_[f(S), g(S)]
		self.H = {}
		self.alpha = np.empty(shape=(0, 2), dtype=np.float64)
		self.subdivision = GeometryCollection()
		self.centroids = np.empty(shape=(0, 2), dtype=np.float64)
		self.templates = {}

	def figure_hilbert(self):
		p = figure_betti(self.H)
		return p

	def compute_firep(self) -> None:
		self.H = bigraded_betti(self.S, f1=self.f, f2=self.g, p=1, xbin=12, ybin=12)

	def prepare_query(self):
		self.alpha = anchors(self.H)
		self.subdivision, self.centroids = arrangement(self.alpha, self.H["x-grades"], self.H["y-grades"])

	def populate_templates(self):
		import gudhi

		self.templates = {}
		for c in self.centroids:
			a, b = c[0], -c[1]
			st = gudhi.simplex_tree.SimplexTree()
			for d in range(self.S.dim() + 1):
				d_simplices = np.array(self.S.faces(d))
				f1 = self.f(d_simplices)
				f2 = self.g(d_simplices)
				fv = push_map(np.c_[f1, f2], a=a, b=b)
				st.insert_batch(d_simplices.T, fv)
			assert not st.make_filtration_non_decreasing()
			st.compute_persistence()
			self.templates[tuple(c)] = st.persistence_pairs()

	def __repr__(self) -> str:
		msg = "Rank Invariant Visualization and Exploration Tool"
		msg += f"# Templates {0}"
		# print(Counter([len(bs) - 1 for bs, ds in dgm]))

	def retrieve_cell_index(self, point: np.ndarray):
		assert len(self.subdivision.geoms) == len(self.centroids)
		pt = Point(*point)
		for i, cell in enumerate(self.subdivision.geoms):
			if cell.contains(pt):
				return i
		return min(enumerate(self.subdivision.geoms), key=lambda p: p[1].distance(pt))[0]

	def select_line(self, a: float, b: float):
		ind = self.retrieve_cell_index([a, -b])
		ind_template = self.templates[tuple(self.centroids[ind])]

		## Replace template values with their persistence values
		fv = push_map(self._f_cross_g, a=a, b=b)
		f_map = dict(zip(self.S, fv))
		f_map[tuple()] = np.inf
		dgm = np.array([(len(bs) - 1, f_map[tuple(sorted(bs))], f_map[tuple(sorted(ds))]) for bs, ds in ind_template])
		return dgm

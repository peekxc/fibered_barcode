from re import template
from bokeh.models import Range1d
import numpy as np
import splex as sx
from rivet import bigraded_betti, anchors, arrangement, figure_betti
from simplextree import SimplexTree
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import gaussian_kde
from bokeh.plotting import figure, show
from bokeh.io import export_svg, export_png, output_notebook
from map2color import map2hex

output_notebook()


def sample_circle(n: int, noise: float = 0.0, outliers: int = 0, seed: int = 1240):
	rng = np.random.default_rng(seed)
	theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
	circle = np.c_[np.cos(theta), np.sin(theta)]
	h_noise = rng.normal(size=(n, 2), loc=0.0, scale=noise)
	s_outliers = rng.uniform(size=(outliers, 2), low=-1, high=+1)
	return np.vstack((circle + h_noise, s_outliers))


## Sample some points
X = sample_circle(150, noise=0.1, outliers=150)
KDE = gaussian_kde(X.T)
Xf_pdf = KDE.pdf(X.T)

## Show the points colored by their densities
p = figure(width=350, height=350)
p.scatter(*X.T, size=8, color=map2hex(Xf_pdf), line_color="black", line_width=1.2)
show(p)

## Form a delaunay complex, bi-filtered by diameter and (co)-density
S = SimplexTree(sx.delaunay_complex(X))
f_diam = sx.flag_filter(pdist(X))
f_dens = sx.lower_star_filter(max(Xf_pdf) - Xf_pdf)

## Get the bigraded betti numbers
H = bigraded_betti(S, f1=f_dens, f2=f_diam, p=1, xbin=12, ybin=12)
p = figure_betti(H)
show(p)

## Get the anchors + the subdivision via point-line duality
alpha = anchors(H)
subdivision, TP = arrangement(alpha, H["x-grades"], H["y-grades"])

## Show the arrangement + it's centroids
from shapely import get_parts

q = figure(width=250, height=250)
for poly in get_parts(subdivision):
	xy = np.array(poly.boundary.coords)
	q.patch(x=xy[:, 0], y=xy[:, 1], line_color="black", fill_color="white")
centroids = np.array([np.array(poly.centroid.coords).flatten() for poly in get_parts(subdivision)])
q.scatter(*centroids.T, color="orange", line_color="black")
show(q)

## Show the template lines
for c in centroids:
	a, b = c[0], -c[1]
	x_vals = np.linspace(-10, 10, 400)
	y_vals = a * x_vals + b
	p.line(x_vals, y_vals, line_width=0.5, color="red")
p.x_range = Range1d(0, 0.22)
p.y_range = Range1d(0, 1.20)
show(p)

## Show a good template line
p = figure_betti(H)
a, b = 4, 0.2
x_vals = np.linspace(-10, 10, 400)
y_vals = a * x_vals + b
p.line(x_vals, y_vals, line_width=0.5, color="red")
p.x_range = Range1d(0, 0.22)
p.y_range = Range1d(0, 1.20)
show(p)


## Compute template barcodes
c = centroids[0]

a = c[0]
b = -c[1]
from rivet import push_map

f1 = f_dens(S)
f2 = f_diam(S)
fv = push_map(np.c_[f1, f2], a=a, b=b)


## Fibered barcode:
## 0. Compute template barcodes
## 1. choose a line via slope/intercept
## 2. Use point line duality to get point in dual space
## 3. Find the template point corresponding to dual point, this gives templates barcode
## 4. Use push_map to get filtration values
## 5. "fit" template barcode to filtration values using pairing

import gudhi

# st = gudhi.simplex_tree.SimplexTree()
# st.insert([0], 0.0)
# st.insert([1], 0.0)
# st.insert([2], 0.0)
# st.insert([0, 1], 1.0)
# st.insert([3], 1.1)
# st.insert([1, 3], 1.2)
# st.insert([2, 3], 1.2)
# st.insert([0, 3], 1.3)
# st.insert([0, 2], 1.4)
# st.insert([1, 2], 1.4)
# st.insert([0, 1, 2], 1.5)
# st.make_filtration_non_decreasing()
# st.compute_persistence()
# st.persistence_pairs()


from collections import Counter

template_points = {}
for c in centroids:
	a, b = c[0], -c[1]
	st = gudhi.simplex_tree.SimplexTree()
	for d in range(S.dim() + 1):
		d_simplices = np.array(S.faces(d))
		f1 = f_dens(d_simplices)
		f2 = f_diam(d_simplices)
		fv = push_map(np.c_[f1, f2], a=a, b=b)
		st.insert_batch(d_simplices.T, fv)
	assert not st.make_filtration_non_decreasing()
	st.compute_persistence()
	dgm = st.persistence_pairs()
	template_points[tuple(c)] = dgm
	print(Counter([len(bs) - 1 for bs, ds in dgm]))


from shapely import Point
from shapely import GeometryCollection


### TODO: speed up with rtree or similar
## https://stackoverflow.com/questions/20297977/looking-for-a-fast-way-to-find-the-polygon-a-point-belongs-to-using-shapely
def retrieve_cell_index(point: np.ndarray, subdivision: GeometryCollection, centroids: np.ndarray):
	assert len(subdivision.geoms) == len(centroids)
	pt = Point(*point)
	for i, cell in enumerate(subdivision.geoms):
		if cell.contains(pt):
			return i
	return min(enumerate(subdivision.geoms), key=lambda p: p[1].distance(pt))[0]


ind = retrieve_cell_index([a, -b], subdivision, centroids)
ind_template = template_points[tuple(centroids[ind])]

## Replace template values with their persistence values
f1 = f_dens(S)
f2 = f_diam(S)
fv = push_map(np.c_[f1, f2], a=a, b=b)
f_map = dict(zip(S, fv))
f_map[tuple()] = np.inf

dgm = np.array([(len(bs) - 1, f_map[tuple(sorted(bs))], f_map[tuple(sorted(ds))]) for bs, ds in ind_template])

from plotting import figure_dgm

show(figure_dgm(dgm[:, 1:]))


from rivet import RIVET

rivet = RIVET(S, f_dens, f_diam)
rivet.compute_firep()
rivet.prepare_query()
rivet.populate_templates()

show(rivet.figure_hilbert())
rivet.fi


# import splex as sx
# from scipy.spatial.distance import pdist, squareform

# X = np.random.uniform(size=(100, 2))

# f_diam = sx.flag_filter(pdist(X))
# f_diam(S)

# ## These all match
# # sx.rips_complex(X, radius=0.25)
# # sx.rips_complex(pdist(X), radius=0.25)
# # sx.rips_complex(squareform(pdist(X)), radius=0.25)

# sx.as_pairwise_dist(squareform(pdist(X)))

# xg, yg = H["x-grades"], H["y-grades"]

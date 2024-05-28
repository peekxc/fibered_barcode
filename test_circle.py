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

np.random.seed(1240)
def sample_circle(n: int, noise: float = 0.0, outliers: int = 0):
  theta = np.linspace(0, 2*np.pi, n, endpoint=False)
  circle = np.c_[np.cos(theta), np.sin(theta)]
  h_noise = np.random.normal(size=(n,2), loc=0.0, scale=noise)
  s_outliers = np.random.uniform(size=(outliers, 2), low=-1, high=+1)
  return np.vstack((circle + h_noise, s_outliers))

## Sample some points
X = sample_circle(500, noise=0.1, outliers=150)
KDE = gaussian_kde(X.T)
Xf_pdf = KDE.pdf(X.T)

## Show the points colored by their densities
p = figure(width=350, height=350)
p.scatter(*X.T, size=8, color=map2hex(Xf_pdf), line_color='black', line_width=1.2)
show(p)

## Form a delaunay complex, bi-filtered by diameter and (co)-density
S = SimplexTree(sx.delaunay_complex(X))
f_diam = sx.flag_filter(pdist(X))
f_dens = sx.lower_star_filter(max(Xf_pdf) - Xf_pdf)

## Get the bigraded betti numbers
H = bigraded_betti(S, f1=f_dens, f2=f_diam, p=1, xbin=12, ybin=12)
show(figure_betti(H))

## Get the anchors + the subdivision via point-line duality  
alpha = anchors(H)
subdivision, TP = arrangement(alpha, H['x-grades'], H['y-grades'])

## Show the arrangement + it's centroids
from shapely import get_parts
q = figure(width=250, height=250)
for poly in get_parts(subdivision):
  xy = np.array(poly.boundary.coords)
  q.patch(x=xy[:,0], y=xy[:,1], line_color='black', fill_color='white')
centroids = np.array([np.array(poly.centroid.coords).flatten() for poly in get_parts(subdivision)])
q.scatter(*centroids.T, color='orange', line_color='black')
show(q)


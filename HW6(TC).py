import numpy as np
import pandas as pd
import altair as alt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import streamlit as st

st.title("K-Means 5 Clusters")

X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state = 1)
df = pd.DataFrame(X, columns = list("ab"))
starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])

i = st.slider("Choose the number of iterations",
              min_value = 1, max_value = 10**2)

kmeans = KMeans(n_clusters = 5, max_iter=i, init=starting_points, n_init = i)
kmeans.fit(X);
df["c"] = kmeans.predict(X)
chart1 = alt.Chart(df).mark_circle().encode(
    x = "a",
    y = "b",
    color = "c:N"
)

df_centers = pd.DataFrame(kmeans.cluster_centers_, columns = list("ab"))

chart_centers = alt.Chart(df_centers).mark_point().encode(
    x = "a",
    y = "b",
    color = alt.value("black"),
    shape = alt.value("diamond"),
)

chart1 + chart_centers
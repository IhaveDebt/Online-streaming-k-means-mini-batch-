#!/usr/bin/env python3
"""
Streaming K-Means (mini-batch) — online centroid updates.
Run: python3 src/streaming_kmeans.py
"""
import random
import math

class StreamingKMeans:
    def __init__(self, k=3, dim=2, lr=0.1):
        self.k = k
        self.dim = dim
        self.lr = lr
        self.centroids = [ [random.random() for _ in range(dim)] for _ in range(k) ]

    def partial_fit(self, x):
        # find closest centroid
        dists = [sum((cx - xi)**2 for cx, xi in zip(c, x)) for c in self.centroids]
        idx = min(range(self.k), key=lambda i:dists[i])
        # update centroid
        c = self.centroids[idx]
        for i in range(self.dim):
            c[i] += self.lr * (x[i] - c[i])

    def predict(self, x):
        dists = [sum((cx - xi)**2 for cx, xi in zip(c, x)) for c in self.centroids]
        return min(range(self.k), key=lambda i:dists[i])

def demo():
    m = StreamingKMeans(k=3, dim=2, lr=0.2)
    # simulate clusters
    for _ in range(1000):
        center = random.choice([(0,0),(5,5),(0,5)])
        x = [random.gauss(center[0],0.6), random.gauss(center[1],0.6)]
        m.partial_fit(x)
    print("Centroids:", m.centroids)
    print("Prediction example:", m.predict([0.1, -0.2]))

if __name__ == "__main__":
    demo()

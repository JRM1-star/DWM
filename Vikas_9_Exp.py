import numpy as np
def hits_algorithm(adj_matrix, max_iter=100, tol=1e-6):
n = adj_matrix.shape[0]
auth = np.ones(n)
hub = np.ones(n)
for _ in range(max_iter):
new_auth = adj_matrix.T @ hub
new_hub = adj_matrix @ new_auth
# Normalize
new_auth = new_auth / np.linalg.norm(new_auth, 2)
new_hub = new_hub / np.linalg.norm(new_hub, 2)
# Check convergence
if np.allclose(auth, new_auth, atol=tol) and np.allclose(hub, new_hub, atol=tol):
break
auth, hub = new_auth, new_hub
return auth, hub
# Example Graph
A = np.array([[0,1,1,0],
[0,0,1,0],
[0,1,0,0],
[0,0,1,0]])
authority, hub = hits_algorithm(A)
print(&quot;Authority Scores:&quot;, authority)
print(&quot;Hub Scores:&quot;, hub)

#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

class Graph
{
public:
    int V;
    vector<vector<int>> adj;

    Graph(int vertices)
    {
        V = vertices;
        adj.resize(V);
    }

    void addEdge(int u, int v)
    {
        // Undirected Graph
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void parallelBFS(int start)
    {
        vector<bool> visited(V, false);
        queue<int> q;
        visited[start] = true;
        q.push(start);
        cout << "Parallel BFS starting from node " << start << ":\n";

        while (!q.empty())
        {
            int levelSize = q.size();
#pragma omp parallel for
            for (int i = 0; i < levelSize; ++i)
            {
                int current;
#pragma omp critical
                {
                    if (!q.empty())
                    {
                        current = q.front();
                        q.pop();
                        cout << "Thread " << omp_get_thread_num() << " visiting " << current << endl;
                    }
                }
#pragma omp parallel for
                for (int j = 0; j < adj[current].size(); ++j)
                {
                    int neighbor = adj[current][j];
#pragma omp critical
                    {
                        if (!visited[neighbor])
                        {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }
    void parallelDFSUtil(int v, vector<bool> &visited)
    {
#pragma omp critical
        {
            visited[v] = true;
            cout << "Thread " << omp_get_thread_num() << " visiting " << v << endl;
        }

        for (int u : adj[v])
        {
            if (!visited[u])
            {
#pragma omp task firstprivate(u)
                {
                    parallelDFSUtil(u, visited);
                }
            }
        }
    }
    void parallelDFS(int start)
    {
        vector<bool> visited(V, false);

        cout << "Parallel DFS starting from node " << start << ":\n";
#pragma omp parallel
        {
#pragma omp single
            {
                parallelDFSUtil(start, visited);
            }
        }
    }
};

int main()
{
    int V;
    cout << "Enter number of nodes: ";
    cin >> V;

    Graph g(V);

    int E;
    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter " << E << " edges (u v):\n";
    for (int i = 0; i < E; ++i)
    {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int startNode;
    cout << "Enter start node for BFS/DFS: ";
    cin >> startNode;

    double start_time = omp_get_wtime();
    g.parallelBFS(startNode);
    double end_time = omp_get_wtime();
    cout << "BFS Execution time: " << (end_time - start_time) << " seconds\n\n";

    start_time = omp_get_wtime();
    g.parallelDFS(startNode);
    end_time = omp_get_wtime();
    cout << "DFS Execution time: " << (end_time - start_time) << " seconds\n";

    return 0;
}
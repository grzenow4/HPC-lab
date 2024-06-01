#include <cassert>
#include <iostream>
#include <map>
#include <math.h>
#include <random>
#include <set>

#include "tbb/tbb.h"

#define WEIGHT_MAX 100

typedef int node_t;
typedef std::pair<node_t, int> edge_t;
typedef std::map<node_t, std::set<edge_t>> graph_t;

void rand_init_graph(graph_t& graph, int node_count, double edge_probability) {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> dis{0, 1};
    for (int i = 0; i < node_count; i++) {
        for (int j = i + 1; j < node_count; j++) {
            if (dis(gen) < edge_probability) {
                int weight = rand() % WEIGHT_MAX;
                graph[i].insert({j, weight});
                graph[j].insert({i, weight});
            }
        }
    }
}

void print_graph(const graph_t& graph) {
    for (auto v: graph) {
        std::cout << v.first << ": [";
        bool first = true;
        for (auto w: v.second) {
            if (!first) {
                std::cout << ", ";
            }
            std::cout << "{" << w.first << ", " << w.second << "}";
            first = false;
        }
        std::cout << "]\n";
    }
}

void dijkstra_seq(graph_t& graph, node_t source, std::map<node_t, int>& dist) {
    std::priority_queue<std::pair<int, node_t>> pq;
    pq.push({0, source});
    dist[source] = 0;

    while (!pq.empty()) {
        auto [current_dist, v] = pq.top();
        pq.pop();

        if (current_dist > dist[v]) {
            continue;
        }

        for (const auto& [u, weight] : graph.at(v)) {
            int new_dist = dist[v] + weight;
            if (new_dist < dist[u]) {
                dist[u] = new_dist;
                pq.push({new_dist, u});
            }
        }
    }
}

void dijkstra_par(graph_t& graph, node_t source, tbb::concurrent_hash_map<node_t, int>& dist) {
    tbb::concurrent_priority_queue<std::pair<int, node_t>> pq;
    {
        tbb::concurrent_hash_map<node_t, int>::accessor a;
        dist.find(a, source);
        a->second = 0;
    }
    pq.push({0, source});

    tbb::parallel_for_each(&source, &source + 1, [&graph, &dist, &pq]
                          (const int& token, tbb::feeder<int>& feeder) {
            std::pair<int, int> p;
            pq.try_pop(p);
            node_t v = p.second;

            int dist_v;
            {
                tbb::concurrent_hash_map<node_t, int>::accessor a;
                dist.find(a, v);
                dist_v = a->second;
            }

            for (auto neigh: graph[v]) {
                int u = neigh.first;
                int weight = neigh.second;

                int dist_u;
                {
                    tbb::concurrent_hash_map<node_t, int>::accessor a;
                    dist.find(a, u);
                    dist_u = a->second;

                    if (dist_u > dist_v + weight) {
                        a->second = dist_v + weight;
                        dist_u = a->second;
                        pq.push({dist_u, u});
                        feeder.add(0);
                    }
                }
            }
        }
    );
}

int main(int argc, char* argv[]) {
    const int node_count = 100;
    graph_t graph;
    rand_init_graph(graph, node_count, 0.5);
    // print_graph(graph);

    node_t source = 0;

    std::map<node_t, int> dist_seq;
    for (int i = 0; i < node_count; i++) {
        dist_seq.insert({i, node_count * WEIGHT_MAX});
    }

    tbb::concurrent_hash_map<node_t, int> dist_par;
    for (int i = 0; i < node_count; i++) {
        dist_par.insert({i, node_count * WEIGHT_MAX});
    }

    dijkstra_seq(graph, source, dist_seq);
    dijkstra_par(graph, source, dist_par);

    for (int i = 0; i < node_count; i++) {
        int seq = dist_seq[i];
        int par;
        {
            tbb::concurrent_hash_map<node_t, int>::accessor a;
            dist_par.find(a, i);
            par = a->second;
        }
        assert(seq == par);
    }
}


/* Levin-Segel stochastic predator-prey model — single realization time-series study.
   Spatial Gillespie / Next Reaction Method on a 1D periodic lattice.
   Records instantaneous snapshots (delta_t=1) after transient for subsystems of
   size 1, 5, 10, and L/2 anchored at site 0.
   Output: 4 plain .dat files. No HDF5, no OpenMP, no MPI.
*/

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <limits>
#include <random>

#include "nrm_heap_header.h"

//----------------------------------------
// Parameter container — populated from input.dat
//----------------------------------------

struct Params {
  long         L;
  double       V;
  double       b, e, p1, p2, d1P, d2P, d1H, d2H;
  double       z, mu1P, mu2P, nu1H, nu2H;
  double       t_transient, T_length;
  unsigned int seed;
};

Params read_params(const std::string& filename) {
  std::map<std::string, double> raw;
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Cannot open parameter file: " << filename << "\n";
    std::exit(1);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    std::string key; double val;
    if (iss >> key >> val) raw[key] = val;
  }

  Params p;
  p.L           = static_cast<long>(raw.at("L"));
  p.V           = raw.at("V");
  p.b           = raw.at("b");
  p.e           = raw.at("e");
  p.p1          = raw.at("p1");
  p.p2          = raw.at("p2");
  p.d1P         = raw.at("d1P");
  p.d2P         = raw.at("d2P");
  p.d1H         = raw.at("d1H");
  p.d2H         = raw.at("d2H");
  p.z           = raw.at("z");
  p.mu1P        = raw.at("mu1P");
  p.mu2P        = raw.at("mu2P");
  p.nu1H        = raw.at("nu1H");
  p.nu2H        = raw.at("nu2H");
  p.t_transient = raw.at("t_transient");
  p.T_length    = raw.at("T_length");
  p.seed        = static_cast<unsigned int>(raw.at("seed"));
  return p;
}

//----------------------------------------
// Simulation class
//----------------------------------------

class Simulation {

private:

  long   L;
  double V;
  double b, e, p1, p2, d1P, d2P, d1H, d2H;
  double z, mu1P, mu2P, nu1H, nu2H;
  double t_transient, t_final;
  unsigned int seed;

  std::vector<long> prey, predator;
  std::vector<std::vector<double>> propensity;
  std::vector<Element> time_heap;
  std::vector<int> SpatialList;

  // Time-series buffers
  std::vector<double> ts_time;
  std::vector<long>   ts_prey1,  ts_pred1;
  std::vector<long>   ts_prey5,  ts_pred5;
  std::vector<long>   ts_prey10, ts_pred10;
  std::vector<long>   ts_preyL2, ts_predL2;

  std::mt19937 gen;
  std::uniform_real_distribution<> real_ran;
  std::uniform_int_distribution<>  int_ran;

  void compute_propensities(long site) {
    propensity[site].clear();

    double TDiffPredator1 = nu1H * predator[site];
    double TDiffPredator2 = nu2H * predator[site] * (predator[site] - 1) / V;
    double TDiffPrey1     = mu1P * prey[site];
    double TDiffPrey2     = mu2P * prey[site] * predator[site] / V;
    double T1 = prey[site] * b;
    double T2 = prey[site] * (prey[site] - 1) * e / V;
    double T3 = prey[site] * predator[site] * p1 / V;
    double T4 = predator[site] * (predator[site] - 1) * d2H / V;

    propensity[site].push_back(TDiffPredator1); // predator diffusion (linear)
    propensity[site].push_back(TDiffPredator2); // predator diffusion (coupled)
    propensity[site].push_back(TDiffPrey1);     // prey diffusion (linear)
    propensity[site].push_back(TDiffPrey2);     // prey diffusion (coupled)
    propensity[site].push_back(T1);             // birth of prey
    propensity[site].push_back(T2);             // assisted birth of prey
    propensity[site].push_back(T3);             // predation
    propensity[site].push_back(T4);             // death due to competition
  }

public:

  Simulation(const Params& p) :
    L(p.L), V(p.V),
    b(p.b), e(p.e), p1(p.p1), p2(p.p2),
    d1P(p.d1P), d2P(p.d2P), d1H(p.d1H), d2H(p.d2H),
    z(p.z), mu1P(p.mu1P), mu2P(p.mu2P), nu1H(p.nu1H), nu2H(p.nu2H),
    t_transient(p.t_transient),
    t_final(p.t_transient + p.T_length),
    seed(p.seed),
    prey(p.L, 0), predator(p.L, 0),
    propensity(p.L),
    time_heap(p.L),
    SpatialList(p.L),
    gen(p.seed),
    real_ran(0.0, 1.0),
    int_ran(0, static_cast<int>(p.L - 1))
  {
    long n_steps = static_cast<long>(p.T_length);
    ts_time.reserve(n_steps);
    ts_prey1.reserve(n_steps);  ts_pred1.reserve(n_steps);
    ts_prey5.reserve(n_steps);  ts_pred5.reserve(n_steps);
    ts_prey10.reserve(n_steps); ts_pred10.reserve(n_steps);
    ts_preyL2.reserve(n_steps); ts_predL2.reserve(n_steps);
  }

  void initialize() {
    double denom = p1 * p1 - d2H * e;
    double rhoP  = (b * d2H) / denom;
    double rhoH  = (b * p1)  / denom;

    double init_prey_den     = rhoP * V;
    double init_predator_den = rhoH * V;

    mu1P = z * mu1P;
    mu2P = z * mu2P;
    nu1H = z * nu1H;
    nu2H = z * nu2H;

    for (int i = 0; i < init_predator_den * L; i++)
      predator[int_ran(gen)]++;
    for (int i = 0; i < init_prey_den * L; i++)
      prey[int_ran(gen)]++;

    for (long site = 0; site < L; site++) {
      compute_propensities(site);
      double rate = std::accumulate(propensity[site].begin(), propensity[site].end(), 0.0);
      std::exponential_distribution<double> exp_dist(rate);
      time_heap[site].even = exp_dist(gen);
      time_heap[site].odd  = site;
    }

    std::make_heap(time_heap.begin(), time_heap.end(), compareEven);

    for (size_t i = 0; i < time_heap.size(); i++)
      SpatialList[time_heap[i].odd] = i;
  }

  void run() {
    long half = L / 2;
    double next_record = t_transient;
    std::vector<int> tag_sites;

    while (time_heap[0].even < t_final) {

      double current_time = time_heap[0].even;

      // record instantaneous snapshot at every integer boundary crossed
      while (current_time > next_record && next_record < t_final) {
        ts_time.push_back(next_record - t_transient);
        ts_prey1.push_back(prey[0]);
        ts_pred1.push_back(predator[0]);
        ts_prey5.push_back(std::accumulate(prey.begin(),     prey.begin()  + 5,    0L));
        ts_pred5.push_back(std::accumulate(predator.begin(), predator.begin() + 5,  0L));
        ts_prey10.push_back(std::accumulate(prey.begin(),     prey.begin()  + 10,   0L));
        ts_pred10.push_back(std::accumulate(predator.begin(), predator.begin() + 10, 0L));
        ts_preyL2.push_back(std::accumulate(prey.begin(),     prey.begin()  + half, 0L));
        ts_predL2.push_back(std::accumulate(predator.begin(), predator.begin() + half, 0L));
        next_record += 1.0;
      }

      long   site = time_heap[0].odd;
      long   ln   = (site - 1 + L) % L;
      long   rn   = (site + 1)     % L;

      double rand_num         = real_ran(gen);
      double total_propensity = std::accumulate(propensity[site].begin(),
                                                propensity[site].end(), 0.0);
      double cumu_prop = 0.0;
      long   i = 0;

      for (const double& elements : propensity[site]) {
        i++;
        if (rand_num * total_propensity >= cumu_prop &&
            rand_num * total_propensity <  cumu_prop + elements)
          break;
        cumu_prop += elements;
      }

      int AffectedNeighbouringSite = L;

      if      (i == 1) { predator[site]--; if (real_ran(gen) < 0.5) { predator[ln]++; AffectedNeighbouringSite = ln; } else { predator[rn]++; AffectedNeighbouringSite = rn; } }
      else if (i == 2) { predator[site]--; if (real_ran(gen) < 0.5) { predator[ln]++; AffectedNeighbouringSite = ln; } else { predator[rn]++; AffectedNeighbouringSite = rn; } }
      else if (i == 3) { prey[site]--;     if (real_ran(gen) < 0.5) { prey[ln]++;     AffectedNeighbouringSite = ln; } else { prey[rn]++;     AffectedNeighbouringSite = rn; } }
      else if (i == 4) { prey[site]--;     if (real_ran(gen) < 0.5) { prey[ln]++;     AffectedNeighbouringSite = ln; } else { prey[rn]++;     AffectedNeighbouringSite = rn; } }
      else if (i == 5) { prey[site]++;                   }
      else if (i == 6) { prey[site]++;                   }
      else if (i == 7) { prey[site]--; predator[site]++; }
      else if (i == 8) { predator[site]--;               }

      if (prey[site] < 0 || predator[site] < 0) {
        std::cout << "code error at site " << site
                  << "  prey=" << prey[site] << "  predator=" << predator[site]
                  << "  event=" << i << "  seed=" << seed << "\n";
        std::ofstream fp("config_error_" + std::to_string(seed) + ".dat");
        for (long j = 0; j < L; j++)
          fp << j << "\t" << prey[j] << "\t" << predator[j] << "\n";
        fp.close();
        break;
      }

      // NRM heap update
      tag_sites.clear();
      tag_sites.push_back(static_cast<int>(site));
      if (AffectedNeighbouringSite != L)
        tag_sites.push_back(AffectedNeighbouringSite);

      for (const int& elem : tag_sites) {
        long s = elem;

        double PreviousPropensity = std::accumulate(propensity[s].begin(),
                                                    propensity[s].end(), 0.0);
        compute_propensities(s);
        double NewPropensity = std::accumulate(propensity[s].begin(),
                                               propensity[s].end(), 0.0);

        if (s == tag_sites[0]) {
          if (NewPropensity != 0.0) {
            std::exponential_distribution<double> exp_dist(NewPropensity);
            time_heap[0].even = current_time + exp_dist(gen);
          } else {
            time_heap[0].even = std::numeric_limits<double>::max();
          }
          heapify(time_heap, 0, SpatialList);
        }
        else if (tag_sites.size() > 1) {
          if (PreviousPropensity != 0.0) {
            time_heap[SpatialList[s]].even = current_time +
              (time_heap[SpatialList[s]].even - current_time) *
              (PreviousPropensity / NewPropensity);
            heapify(time_heap, SpatialList[s], SpatialList);
          }
          else if (PreviousPropensity == 0.0 && NewPropensity != 0.0) {
            std::exponential_distribution<double> exp_dist(NewPropensity);
            time_heap[SpatialList[s]].even = current_time + exp_dist(gen);
            heapify(time_heap, SpatialList[s], SpatialList);
          }
        }
      }

    } // end NRM loop
  }

  void write_output() {
    auto write_file = [&](const std::string& fname,
                          const std::vector<long>& prey_ts,
                          const std::vector<long>& pred_ts) {
      std::ofstream f(fname);
      f << "# time  prey  predator\n";
      for (size_t k = 0; k < ts_time.size(); k++)
        f << ts_time[k] << "\t" << prey_ts[k] << "\t" << pred_ts[k] << "\n";
      std::cout << "Wrote " << ts_time.size() << " rows to " << fname << "\n";
    };

    write_file("timeseries_size1.dat",  ts_prey1,  ts_pred1);
    write_file("timeseries_size5.dat",  ts_prey5,  ts_pred5);
    write_file("timeseries_size10.dat", ts_prey10, ts_pred10);
    write_file("timeseries_sizeL2.dat", ts_preyL2, ts_predL2);
  }
};

//----------------------------------------

int main() {
  Params params = read_params("input_single.dat");

  if (params.L < 10) {
    std::cerr << "L must be >= 10 for all subsystem sizes to fit.\n";
    return 1;
  }

  std::cout << "Single realization: L=" << params.L
            << "  V=" << params.V
            << "  t_transient=" << params.t_transient
            << "  T_length=" << params.T_length
            << "  seed=" << params.seed << "\n";

  Simulation sim(params);
  sim.initialize();
  sim.run();
  sim.write_output();

  return 0;
}

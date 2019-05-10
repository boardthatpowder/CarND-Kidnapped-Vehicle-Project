/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;
using std::numeric_limits;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  
  // Random Gaussian noise...
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Initialize all weights to 1...
  weights.resize(num_particles, 1);
    
  // Initialize all particles
  particles.resize(num_particles);
  for (auto &p: particles) {
    p.id=0;
  	p.x=dist_x(gen);
  	p.y=dist_y(gen);
  	p.theta=dist_theta(gen);
    p.weight=1.0;
  }
  
  // mark as initialized
  is_initialized=true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Random Gaussian noise...
  default_random_engine gen;

  for (auto &p: particles) {
    
	// Add measurements to each particle using Motion Model formula...
    if (yaw_rate==0) {
      p.x=p.x + (velocity*delta_t) * cos(p.theta);
      p.y=p.y + (velocity*delta_t) * sin(p.theta);
      // no change to theta
    } else {
      p.x=p.x + (velocity/yaw_rate) * (sin(p.theta + (yaw_rate*delta_t)) - sin(p.theta));
      p.y=p.y + (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + (yaw_rate*delta_t)));
      p.theta=p.theta+(yaw_rate*delta_t);
    }
    
    // add Random Gaussian noise...
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);
    p.x=dist_x(gen);
    p.y=dist_y(gen);
    p.theta=dist_theta(gen);
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (auto &o: observations) {
     double shortest_distance = numeric_limits<double>::max();
    for (unsigned i=0; i<predicted.size(); i++) {
      double distance = dist(predicted[i].x, predicted[i].y, o.x, o.y);
      if (distance<shortest_distance) {
        shortest_distance=distance;
        o.id=i;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  
  for (unsigned int i=0; i<particles.size(); ++i) {
    
    auto &p = particles[i];
    
    // we're only interested in map landmarks that are within sensor range, therefore ignore the rest
  	std::vector<LandmarkObs> landmarksInRange;
    for (auto &lm: map_landmarks.landmark_list) {
      if (dist(p.x, p.y, lm.x_f, lm.y_f) <= sensor_range) {
        // only include these...
		landmarksInRange.push_back({lm.id_i, lm.x_f, lm.y_f});
      }
    }
    
    // transform car sensor landmark observations from vehicle coordinate system to map coordinate system
    vector<LandmarkObs> transformedObservations;
  	for (auto &o: observations) {
      LandmarkObs transformed;
      transformed.id = o.id;
      transformed.x= (cos(p.theta)*o.x) - (sin(p.theta)*o.y) + p.x;
      transformed.y= (sin(p.theta)*o.x) + (cos(p.theta)*o.y) + p.y;
      transformedObservations.push_back(transformed);
    }
                                        
	// associate transformed observations with nearest landmarks (within sensor range) on map
	dataAssociation(transformedObservations, landmarksInRange);
    
    // update particle weights by applying Gaussian probability density function for each measurement
    double total_weight=1.0;
  	for (auto &lm: landmarksInRange) {
      auto associated=transformedObservations[lm.id];
      double x=associated.x;
      double y=associated.y;
      double sigma_x=std_landmark[0];
      double sigma_y=std_landmark[1];
      double mu_x=lm.x;
      double mu_y=lm.y;
      
      double gauss_norm = 1 / (2 * M_PI * sigma_x * sigma_y);
      double exponent = (pow(x - mu_x, 2) / (2 * pow(sigma_x, 2)))
               + (pow(y - mu_y, 2) / (2 * pow(sigma_y, 2)));
      double weight = gauss_norm * exp(-exponent);
      total_weight*=weight;
    }
    p.weight=total_weight;
	weights[i]=p.weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  default_random_engine gen;
  discrete_distribution<int> d(weights.begin(), weights.end());
  
  vector<Particle> p2s;
  for (unsigned int i=0; i<particles.size(); ++i) {
 	// randomly choose a particle with probablity proportional to its weight
    p2s.push_back(particles[d(gen)]);
  }

  particles=p2s;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
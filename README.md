# Particle Filter C++ 

High-performance Monte Carlo Localization (MCL) for robot navigation. 

Part of the F1TENTH autonomous racing system - provides localization for path planning and control.

## Quick Start

```bash
# Build (from workspace root)
colcon build --packages-select particle_filter_cpp --symlink-install
source install/setup.bash

# For F1TENTH Gym Simulation
ros2 launch particle_filter_cpp localize_sim_launch.py

# For Real F1TENTH Car with SLAM Map
ros2 launch particle_filter_cpp localize_slam_launch.py

# Generic launch (configure manually)
ros2 launch particle_filter_cpp localize_launch.py
```

## Configuration Files

### 1. `config/localize_sim.yaml` - Simulation Config
**Optimized for F1TENTH Gym simulation:**
- Lower motion noise (cleaner simulation physics)
- Higher `z_hit` weight (less sensor noise)
- Default topics: `/ego_racecar/odom`
- Default map: `Spielberg_map`

### 2. `config/localize_slam.yaml` - Real Car Config  
**Tuned for real F1TENTH hardware:**
- Higher motion noise (real-world uncertainties)
- Higher `z_rand` weight (sensor noise compensation)
- Default topics: `/odom`
- Default map: `map_1753950572`

### 3. `config/localize.yaml` - Generic Config
**General purpose configuration** - manually adjust as needed

**Key Parameters:**
- `max_particles`: 4000 (default)
- `max_range`: 5.0 meters  
- `motion_dispersion_*`: Noise parameters
- `z_hit/short/max/rand`: Sensor model weights

## Launch Files

### 1. Simulation Launch (`localize_sim_launch.py`)
**Default Settings:**
- Map: `Spielberg_map` 
- Odom Topic: `/ego_racecar/odom`
- Simulation Time: `True`

### 2. SLAM Map Launch (`localize_slam_launch.py`)
**Default Settings:**
- Map: `map_1753950572` (SLAM generated)
- Odom Topic: `/odom` (real F1TENTH car)
- Simulation Time: `False`

### 3. Generic Launch (`localize_launch.py`)
**Configurable via config file**

## Launch Options

```bash
# Custom map
ros2 launch particle_filter_cpp localize_sim_launch.py map_name:=levine

# Without RViz  
ros2 launch particle_filter_cpp localize_slam_launch.py use_rviz:=false

# Custom topics
ros2 launch particle_filter_cpp localize_sim_launch.py \
    scan_topic:=/custom_scan odom_topic:=/custom_odom
```

## Key Topics

**Subscribes:**
- `/scan` - Laser scan data
- `/ego_racecar/odom` - Odometry data  
- `/initialpose` - Initial pose from RViz

**Publishes:**
- `/pf/viz/particles` - Particle cloud
- `/pf/viz/inferred_pose` - Estimated pose
- `/tf` - Map to laser transform

## How MCL Works

Monte Carlo Localization uses particles to estimate robot pose:

1. **Prediction**: Move particles based on odometry + noise
2. **Update**: Weight particles by laser scan likelihood  
3. **Resampling**: Keep particles with higher weights
4. **Estimation**: Compute weighted average as robot pose

```
Particles → Motion Model → Sensor Model → Resampling → Pose Estimate
   ↑                                                        ↓
   └─────────────── Repeat every update ──────────────────┘
```

## Code Architecture

```cpp
class ParticleFilter {
    // Core MCL algorithm 
    void MCL(action, observation);
    void motion_model(particles, action);     // Add noise to particle motion
    void sensor_model(particles, scan);       // Weight by scan likelihood
    Eigen::Vector3d expected_pose();          // Weighted average
    
    // ROS interface
    void odomCB(odom_msg);                   // Triggers MCL update
    void lidarCB(scan_msg);                  // Stores scan data
    void publish_tf(pose);                   // Publishes results
    
    // State
    Eigen::MatrixXd particles_;              // [N x 3] particle poses
    std::vector<double> weights_;            // Particle weights
    Eigen::MatrixXd sensor_model_table_;     // Pre-computed lookup table
};
```

**Key Data Flow:**
1. Odometry → `odomCB()` → `MCL()` → Pose estimate
2. Laser → `lidarCB()` → Store for next MCL update
3. MCL → `publish_tf()` → ROS topics

## Implementation Notes

- **Vectorized Operations**: Uses Eigen for fast matrix operations  
- **Pre-computed Sensor Model**: Lookup table for fast likelihood computation
- **Simple Ray Casting**: Basic implementation (RangeLibc optional for speed)
- **Memory Optimized**: Pre-allocated arrays, minimal runtime allocation

## Performance

Expect ~10x faster execution compared to Python version due to:
- Compiled C++ vs interpreted Python
- Optimized Eigen matrix operations  
- Reduced memory allocation overhead
- Vectorized particle operations

Built for F1TENTH racing simulation and real-time robotic navigation.

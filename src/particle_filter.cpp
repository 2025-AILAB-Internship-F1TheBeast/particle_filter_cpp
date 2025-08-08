// ================================================================================================
// PARTICLE FILTER IMPLEMENTATION - Monte Carlo Localization (MCL)
// ================================================================================================
// Implements complete MCL algorithm with:
// - Multinomial resampling
// - Velocity motion model with Gaussian noise
// - 4-component beam sensor model with lookup table optimization
// - Real-time ray casting for range simulation
// ================================================================================================

#include "particle_filter_cpp/particle_filter.hpp"
#include "particle_filter_cpp/utils.hpp"
#include <algorithm>
#include <angles/angles.h>
#include <chrono>
#include <cmath>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <numeric>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace particle_filter_cpp
{

// --------------------------------- CONSTRUCTOR & INITIALIZATION ---------------------------------
ParticleFilter::ParticleFilter(const rclcpp::NodeOptions &options)
    : Node("particle_filter", options), rng_(std::random_device{}()), uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0)
{
    // ROS2 parameter declarations (defaults loaded from config file)
    this->declare_parameter("angle_step", 18);
    this->declare_parameter("max_particles", 4000);
    this->declare_parameter("max_viz_particles", 60);
    this->declare_parameter("squash_factor", 2.2);
    this->declare_parameter("max_range", 12.0);
    this->declare_parameter("theta_discretization", 112);
    this->declare_parameter("range_method", "rmgpu");
    this->declare_parameter("rangelib_variant", 1);
    this->declare_parameter("fine_timing", 0);
    this->declare_parameter("publish_odom", true);
    this->declare_parameter("viz", true);
    this->declare_parameter("z_short", 0.01);
    this->declare_parameter("z_max", 0.07);
    this->declare_parameter("z_rand", 0.12);
    this->declare_parameter("z_hit", 0.80);
    this->declare_parameter("sigma_hit", 8.0);
    this->declare_parameter("motion_dispersion_x", 0.05);
    this->declare_parameter("motion_dispersion_y", 0.025);
    this->declare_parameter("motion_dispersion_theta", 0.25);
    this->declare_parameter("lidar_offset_x", 0.0);
    this->declare_parameter("lidar_offset_y", 0.0);
    this->declare_parameter("wheelbase", 0.325);
    this->declare_parameter("scan_topic", "/scan");
    this->declare_parameter("odom_topic", "/odom");

    // Retrieve parameter values
    ANGLE_STEP = this->get_parameter("angle_step").as_int();
    MAX_PARTICLES = this->get_parameter("max_particles").as_int();
    MAX_VIZ_PARTICLES = this->get_parameter("max_viz_particles").as_int();
    INV_SQUASH_FACTOR = 1.0 / this->get_parameter("squash_factor").as_double();
    MAX_RANGE_METERS = this->get_parameter("max_range").as_double();
    THETA_DISCRETIZATION = this->get_parameter("theta_discretization").as_int();
    WHICH_RM = this->get_parameter("range_method").as_string();
    RANGELIB_VAR = this->get_parameter("rangelib_variant").as_int();
    SHOW_FINE_TIMING = this->get_parameter("fine_timing").as_int() > 0;
    PUBLISH_ODOM = this->get_parameter("publish_odom").as_bool();
    DO_VIZ = this->get_parameter("viz").as_bool();

    // 4-component sensor model parameters: Z_HIT + Z_SHORT + Z_MAX + Z_RAND = 1.0
    Z_SHORT = this->get_parameter("z_short").as_double();
    Z_MAX = this->get_parameter("z_max").as_double();
    Z_RAND = this->get_parameter("z_rand").as_double();
    Z_HIT = this->get_parameter("z_hit").as_double();
    SIGMA_HIT = this->get_parameter("sigma_hit").as_double();

    // Process noise for motion model: σ_x, σ_y, σ_θ
    MOTION_DISPERSION_X = this->get_parameter("motion_dispersion_x").as_double();
    MOTION_DISPERSION_Y = this->get_parameter("motion_dispersion_y").as_double();
    MOTION_DISPERSION_THETA = this->get_parameter("motion_dispersion_theta").as_double();

    // LiDAR sensor frame offset and robot geometry
    LIDAR_OFFSET_X = this->get_parameter("lidar_offset_x").as_double();
    LIDAR_OFFSET_Y = this->get_parameter("lidar_offset_y").as_double();
    WHEELBASE = this->get_parameter("wheelbase").as_double();

    // System state initialization
    MAX_RANGE_PX = 0;
    odometry_data_ = Eigen::Vector3d::Zero();
    iters_ = 0;
    map_initialized_ = false;
    lidar_initialized_ = false;
    odom_initialized_ = false;
    first_sensor_update_ = true;
    current_speed_ = 0.0;

    // Particle filter core: N particles with uniform weights
    particles_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);
    weights_.resize(MAX_PARTICLES, 1.0 / MAX_PARTICLES);
    particle_indices_.resize(MAX_PARTICLES);
    std::iota(particle_indices_.begin(), particle_indices_.end(), 0);

    // Cache for motion model
    local_deltas_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);

    // ROS2 publishers for visualization and navigation
    if (DO_VIZ)
    {
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 1);
        particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
        pub_fake_scan_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/pf/viz/fake_scan", 1);
        rect_pub_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>("/pf/viz/poly1", 1);
    }

    if (PUBLISH_ODOM)
    {
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", 1);
    }

    // Initialize TF broadcaster
    pub_tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Sensor data and user input subscribers
    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        this->get_parameter("scan_topic").as_string(), 1,
        std::bind(&ParticleFilter::lidarCB, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        this->get_parameter("odom_topic").as_string(), 1,
        std::bind(&ParticleFilter::odomCB, this, std::placeholders::_1));

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1, std::bind(&ParticleFilter::clicked_pose, this, std::placeholders::_1));

    click_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/clicked_point", 1, std::bind(&ParticleFilter::clicked_point, this, std::placeholders::_1));

    // Initialize map service client
    map_client_ = this->create_client<nav_msgs::srv::GetMap>("/map_server/map");

    // Get the map
    get_omap();
    initialize_global();

    RCLCPP_INFO(this->get_logger(), "Finished initializing, waiting on messages...");
}

// --------------------------------- MAP LOADING & PREPROCESSING ---------------------------------
void ParticleFilter::get_omap()
{
    RCLCPP_INFO(this->get_logger(), "Requesting map from map server...");

    while (!map_client_->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok())
            return;
        RCLCPP_INFO(this->get_logger(), "Get map service not available, waiting...");
    }

    auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();
    auto future = map_client_->async_send_request(request);

    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS)
    {

        map_msg_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(future.get()->map);
        map_resolution_ = map_msg_->info.resolution;
        map_origin_ = Eigen::Vector3d(map_msg_->info.origin.position.x, map_msg_->info.origin.position.y,
                                      quaternion_to_angle(map_msg_->info.origin.orientation));

        MAX_RANGE_PX = static_cast<int>(MAX_RANGE_METERS / map_resolution_);

        RCLCPP_INFO(this->get_logger(), "Initializing range method: %s", WHICH_RM.c_str());

        // Extract free space (occupancy = 0) for particle initialization
        int height = map_msg_->info.height;
        int width = map_msg_->info.width;
        permissible_region_ = Eigen::MatrixXi::Zero(height, width);

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int idx = i * width + j;
                if (idx < static_cast<int>(map_msg_->data.size()) && map_msg_->data[idx] == 0)
                {
                    permissible_region_(i, j) = 1; // permissible
                }
            }
        }

        map_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "Done loading map");

        // Generate lookup table for fast sensor model evaluation
        precompute_sensor_model();
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get map from map server");
    }
}

// --------------------------------- SENSOR MODEL PRECOMPUTATION ---------------------------------
void ParticleFilter::precompute_sensor_model()
{
    RCLCPP_INFO(this->get_logger(), "Precomputing sensor model");

    // Safety check for map resolution
    if (map_resolution_ <= 0.0)
    {
        RCLCPP_ERROR(this->get_logger(), "Invalid map resolution: %.6f. Cannot precompute sensor model.",
                     map_resolution_);
        return;
    }

    int table_width = MAX_RANGE_PX + 1;
    RCLCPP_INFO(this->get_logger(), "MAX_RANGE_METERS: %.2f, map_resolution_: %.6f", MAX_RANGE_METERS, map_resolution_);
    RCLCPP_INFO(this->get_logger(), "MAX_RANGE_PX: %d, table_width: %d", MAX_RANGE_PX, table_width);

    // Safety check for reasonable table size
    if (table_width <= 0 || table_width > 10000)
    {
        RCLCPP_ERROR(this->get_logger(), "Invalid table_width: %d. Cannot allocate sensor model table.", table_width);
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Attempting to allocate %dx%d matrix (%zu MB)", table_width, table_width,
                (table_width * table_width * sizeof(double)) / (1024 * 1024));

    sensor_model_table_ = Eigen::MatrixXd::Zero(table_width, table_width);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build (observed_range × expected_range) probability lookup table
    for (int d = 0; d < table_width; ++d)  // d = expected range (raycast result)
    {
        double norm = 0.0;

        for (int r = 0; r < table_width; ++r)  // r = observed range (sensor reading)
        {
            double prob = 0.0;
            double z = static_cast<double>(r - d);

            // Z_HIT: Gaussian around expected range (correct measurement)
            prob += Z_HIT * std::exp(-(z * z) / (2.0 * SIGMA_HIT * SIGMA_HIT)) / (SIGMA_HIT * std::sqrt(2.0 * M_PI));

            // Z_SHORT: Exponential for early obstacles
            if (r < d)
            {
                prob += 2.0 * Z_SHORT * (d - r) / static_cast<double>(d);
            }

            // Z_MAX: Delta function at maximum range
            if (r == MAX_RANGE_PX)
            {
                prob += Z_MAX;
            }

            // Z_RAND: Uniform distribution for noise
            if (r < MAX_RANGE_PX)
            {
                prob += Z_RAND * 1.0 / static_cast<double>(MAX_RANGE_PX);
            }

            norm += prob;
            sensor_model_table_(r, d) = prob;
        }

        // Normalize to form valid probability distribution
        if (norm > 0)
        {
            sensor_model_table_.col(d) /= norm;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_INFO(this->get_logger(), "Sensor model precomputed in %ld ms", duration.count());
}

// --------------------------------- SENSOR CALLBACKS ---------------------------------
void ParticleFilter::lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    if (laser_angles_.empty())
    {
        RCLCPP_INFO(this->get_logger(), "...Received first LiDAR message");

        // First message: extract scan parameters and downsample
        laser_angles_.resize(msg->ranges.size());
        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            laser_angles_[i] = msg->angle_min + i * msg->angle_increment;
        }

        // Create downsampled angles
        for (size_t i = 0; i < laser_angles_.size(); i += ANGLE_STEP)
        {
            downsampled_angles_.push_back(laser_angles_[i]);
        }

        // Initialize visualization arrays
        viz_queries_ = Eigen::MatrixXd::Zero(downsampled_angles_.size(), 3);
        viz_ranges_.resize(downsampled_angles_.size());

        RCLCPP_INFO(this->get_logger(), "Downsampled to %zu angles", downsampled_angles_.size());
    }

    // Extract every ANGLE_STEP-th measurement for efficiency
    downsampled_ranges_.clear();
    for (size_t i = 0; i < msg->ranges.size(); i += ANGLE_STEP)
    {
        downsampled_ranges_.push_back(msg->ranges[i]);
    }

    lidar_initialized_ = true;
}

void ParticleFilter::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    Eigen::Vector3d position(msg->pose.pose.position.x, msg->pose.pose.position.y,
                             quaternion_to_angle(msg->pose.pose.orientation));

    current_speed_ = msg->twist.twist.linear.x;

    if (last_pose_.norm() > 0)
    { // Check if we have a previous pose
        // Transform global displacement to robot-local coordinates
        Eigen::Matrix2d rot = rotation_matrix(-last_pose_[2]);
        Eigen::Vector2d delta = position.head<2>() - last_pose_.head<2>();
        Eigen::Vector2d local_delta = rot * delta;

        odometry_data_ = Eigen::Vector3d(local_delta[0], local_delta[1], position[2] - last_pose_[2]);
        last_pose_ = position;
        last_stamp_ = msg->header.stamp;
        odom_initialized_ = true;
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "...Received first Odometry message");
        last_pose_ = position;
    }

    // Trigger MCL update cycle
    update();
}

// --------------------------------- INTERACTIVE INITIALIZATION ---------------------------------
void ParticleFilter::clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    Eigen::Vector3d pose(msg->pose.pose.position.x, msg->pose.pose.position.y,
                         quaternion_to_angle(msg->pose.pose.orientation));
    initialize_particles_pose(pose);
}

void ParticleFilter::clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr /*msg*/)
{
    initialize_global();
}

// --------------------------------- PARTICLE INITIALIZATION ---------------------------------
void ParticleFilter::initialize_particles_pose(const Eigen::Vector3d &pose)
{
    RCLCPP_INFO(this->get_logger(), "SETTING POSE");
    RCLCPP_INFO(this->get_logger(), "Position: [%.3f, %.3f]", pose[0], pose[1]);

    std::lock_guard<std::mutex> lock(state_lock_);

    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

    // Gaussian distribution around clicked pose
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        particles_(i, 0) = pose[0] + normal_dist_(rng_) * 0.5;  // σ_x = 0.5m
        particles_(i, 1) = pose[1] + normal_dist_(rng_) * 0.5;  // σ_y = 0.5m
        particles_(i, 2) = pose[2] + normal_dist_(rng_) * 0.4;  // σ_θ = 0.4rad
    }
}

void ParticleFilter::initialize_global()
{
    if (!map_initialized_)
        return;

    RCLCPP_INFO(this->get_logger(), "GLOBAL INITIALIZATION");

    std::lock_guard<std::mutex> lock(state_lock_);

    // Extract all free space cells for uniform sampling
    std::vector<std::pair<int, int>> permissible_positions;
    for (int i = 0; i < permissible_region_.rows(); ++i)
    {
        for (int j = 0; j < permissible_region_.cols(); ++j)
        {
            if (permissible_region_(i, j) == 1)
            {
                permissible_positions.emplace_back(i, j);
            }
        }
    }

    if (permissible_positions.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No permissible positions found in map!");
        return;
    }

    // Uniform sampling over free space
    std::uniform_int_distribution<int> pos_dist(0, permissible_positions.size() - 1);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        int idx = pos_dist(rng_);
        auto pos = permissible_positions[idx];

        // Grid to world coordinate transformation
        particles_(i, 0) = pos.second * map_resolution_ + map_origin_[0]; // x = j * resolution + origin_x
        particles_(i, 1) = pos.first * map_resolution_ + map_origin_[1];  // y = i * resolution + origin_y
        particles_(i, 2) = angle_dist(rng_);
    }

    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

    RCLCPP_INFO(this->get_logger(), "Initialized %d particles from %zu permissible positions", MAX_PARTICLES,
                permissible_positions.size());
}

// --------------------------------- MCL ALGORITHM CORE ---------------------------------
void ParticleFilter::motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action)
{
    // Apply motion transformation: local → global coordinates
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        double cos_theta = std::cos(proposal_dist(i, 2));
        double sin_theta = std::sin(proposal_dist(i, 2));

        local_deltas_(i, 0) = cos_theta * action[0] - sin_theta * action[1];
        local_deltas_(i, 1) = sin_theta * action[0] + cos_theta * action[1];
        local_deltas_(i, 2) = action[2];
    }

    proposal_dist += local_deltas_;

    // Add Gaussian process noise
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        proposal_dist(i, 0) += normal_dist_(rng_) * MOTION_DISPERSION_X;
        proposal_dist(i, 1) += normal_dist_(rng_) * MOTION_DISPERSION_Y;
        proposal_dist(i, 2) += normal_dist_(rng_) * MOTION_DISPERSION_THETA;
    }
}

void ParticleFilter::sensor_model(const Eigen::MatrixXd &proposal_dist, const std::vector<float> &obs,
                                  std::vector<double> &weights)
{
    int num_rays = downsampled_angles_.size();

    // First-time array allocation for ray casting
    if (first_sensor_update_)
    {
        queries_ = Eigen::MatrixXd::Zero(num_rays * MAX_PARTICLES, 3);
        ranges_.resize(num_rays * MAX_PARTICLES);
        tiled_angles_.clear();
        for (int i = 0; i < MAX_PARTICLES; ++i)
        {
            tiled_angles_.insert(tiled_angles_.end(), downsampled_angles_.begin(), downsampled_angles_.end());
        }
        first_sensor_update_ = false;
    }

    // Generate ray queries: transform base_link pose to LiDAR frame
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        // Transform particle pose from base_link to LiDAR world position
        Eigen::Vector2d lidar_world_pos = transform_to_lidar_frame(proposal_dist.row(i).transpose());
        
        for (int j = 0; j < num_rays; ++j)
        {
            int idx = i * num_rays + j;
            queries_(idx, 0) = lidar_world_pos[0];  // LiDAR X position in world frame
            queries_(idx, 1) = lidar_world_pos[1];  // LiDAR Y position in world frame
            queries_(idx, 2) = proposal_dist(i, 2) + downsampled_angles_[j];  // Ray angle from LiDAR
        }
    }

    // Batch ray casting: simulate LiDAR measurements
    ranges_ = calc_range_many(queries_);

    // Convert to pixel units and compute particle weights
    std::vector<float> obs_px(obs.size());
    std::vector<float> ranges_px(ranges_.size());

    for (size_t i = 0; i < obs.size(); ++i)
    {
        obs_px[i] = obs[i] / map_resolution_;
        if (obs_px[i] > MAX_RANGE_PX)
            obs_px[i] = MAX_RANGE_PX;
    }

    for (size_t i = 0; i < ranges_.size(); ++i)
    {
        ranges_px[i] = ranges_[i] / map_resolution_;
        if (ranges_px[i] > MAX_RANGE_PX)
            ranges_px[i] = MAX_RANGE_PX;
    }

    // Likelihood calculation using lookup table
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        double weight = 1.0;
        for (int j = 0; j < num_rays; ++j)
        {
            int obs_idx = static_cast<int>(std::round(obs_px[j]));
            int range_idx = static_cast<int>(std::round(ranges_px[i * num_rays + j]));

            obs_idx = std::max(0, std::min(obs_idx, MAX_RANGE_PX));
            range_idx = std::max(0, std::min(range_idx, MAX_RANGE_PX));

            weight *= sensor_model_table_(obs_idx, range_idx);
        }
        weights[i] = std::pow(weight, INV_SQUASH_FACTOR);
    }
}

// --------------------------------- RAY CASTING ---------------------------------
std::vector<float> ParticleFilter::calc_range_many(const Eigen::MatrixXd &queries)
{
    std::vector<float> results(queries.rows());

    for (int i = 0; i < queries.rows(); ++i)
    {
        results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2));
    }

    return results;
}

float ParticleFilter::cast_ray(double x, double y, double angle)
{
    if (!map_initialized_)
        return MAX_RANGE_METERS;

    double dx = std::cos(angle) * map_resolution_;
    double dy = std::sin(angle) * map_resolution_;

    double current_x = x;
    double current_y = y;

    for (int step = 0; step < MAX_RANGE_PX; ++step)
    {
        current_x += dx;
        current_y += dy;

        // World to grid coordinate transformation
        int grid_x = static_cast<int>((current_x - map_origin_[0]) / map_resolution_);
        int grid_y = static_cast<int>((current_y - map_origin_[1]) / map_resolution_);

        // Map boundary collision
        if (grid_x < 0 || grid_x >= static_cast<int>(map_msg_->info.width) || grid_y < 0 ||
            grid_y >= static_cast<int>(map_msg_->info.height))
        {
            return step * map_resolution_;
        }

        // Obstacle collision detection
        int map_idx = grid_y * map_msg_->info.width + grid_x;
        if (map_idx >= 0 && map_idx < static_cast<int>(map_msg_->data.size()))
        {
            if (map_msg_->data[map_idx] > 50)
            { // Occupied
                return step * map_resolution_;
            }
        }
    }

    return MAX_RANGE_METERS;
}

void ParticleFilter::MCL(const Eigen::Vector3d &action, const std::vector<float> &observation)
{
    // Step 1: Multinomial resampling based on weights
    std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());
    Eigen::MatrixXd proposal_distribution(MAX_PARTICLES, 3);

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        int idx = particle_dist(rng_);
        proposal_distribution.row(i) = particles_.row(idx);
    }

    // Step 2: Motion prediction with noise
    motion_model(proposal_distribution, action);

    // Step 3: Sensor likelihood evaluation
    sensor_model(proposal_distribution, observation, weights_);

    // Step 4: Weight normalization for probability distribution
    double sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (sum_weights > 0)
    {
        for (double &w : weights_)
        {
            w /= sum_weights;
        }
    }

    // Step 5: Update particle set
    particles_ = proposal_distribution;
}

Eigen::Vector3d ParticleFilter::expected_pose()
{
    Eigen::Vector3d pose = Eigen::Vector3d::Zero();
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        pose += weights_[i] * particles_.row(i).transpose();
    }
    return pose;
}

// --------------------------------- MAIN UPDATE LOOP ---------------------------------
void ParticleFilter::update()
{
    if (!lidar_initialized_ || !odom_initialized_ || !map_initialized_)
    {
        return;
    }

    if (state_lock_.try_lock())
    {
        ++iters_;

        auto observation = downsampled_ranges_;
        auto action = odometry_data_;
        odometry_data_ = Eigen::Vector3d::Zero();

        // Execute complete MCL cycle
        MCL(action, observation);

        // Final pose estimate: weighted mean
        inferred_pose_ = expected_pose();

        state_lock_.unlock();

        // Output to navigation stack and visualization
        publish_tf(inferred_pose_, last_stamp_);

        if (iters_ % 10 == 0)
        {
            RCLCPP_INFO(this->get_logger(), "MCL iteration %d, pose: (%.3f, %.3f, %.3f)", iters_, inferred_pose_[0],
                        inferred_pose_[1], inferred_pose_[2]);
        }

        visualize();
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Concurrency error avoided");
    }
}

// --------------------------------- OUTPUT & VISUALIZATION ---------------------------------
void ParticleFilter::publish_tf(const Eigen::Vector3d &pose, const rclcpp::Time &stamp)
{
    // Calculate map → odom transform: T_map_odom = T_map_base * T_base_odom
    // where T_base_odom = inverse(T_odom_base)
    Eigen::Vector3d map_to_odom_pose = pose - last_pose_;  // Difference between particle filter estimate and odom
    
    // Publish map → odom transform for localization
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp.nanoseconds() > 0 ? stamp : this->get_clock()->now();
    t.header.frame_id = "map";
    t.child_frame_id = "odom";
    t.transform.translation.x = map_to_odom_pose[0];
    t.transform.translation.y = map_to_odom_pose[1];
    t.transform.translation.z = 0.0;
    t.transform.rotation = angle_to_quaternion(map_to_odom_pose[2]);

    pub_tf_->sendTransform(t);

    // Optional navigation odometry output
    if (PUBLISH_ODOM && odom_pub_)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = this->get_clock()->now();
        odom.header.frame_id = "/map";
        odom.pose.pose.position.x = pose[0];
        odom.pose.pose.position.y = pose[1];
        odom.pose.pose.orientation = angle_to_quaternion(pose[2]);
        odom.twist.twist.linear.x = current_speed_;
        odom_pub_->publish(odom);
    }
}

void ParticleFilter::visualize()
{
    if (!DO_VIZ)
        return;

    // RViz pose visualization
    if (pose_pub_ && pose_pub_->get_subscription_count() > 0)
    {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = this->get_clock()->now();
        ps.header.frame_id = "/map";
        ps.pose.position.x = inferred_pose_[0];
        ps.pose.position.y = inferred_pose_[1];
        ps.pose.orientation = angle_to_quaternion(inferred_pose_[2]);
        pose_pub_->publish(ps);
    }

    // RViz particle cloud (downsampled for performance)
    if (particle_pub_ && particle_pub_->get_subscription_count() > 0)
    {
        if (MAX_PARTICLES > MAX_VIZ_PARTICLES)
        {
            // Weighted downsampling to preserve distribution shape
            std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());
            Eigen::MatrixXd viz_particles(MAX_VIZ_PARTICLES, 3);

            for (int i = 0; i < MAX_VIZ_PARTICLES; ++i)
            {
                int idx = particle_dist(rng_);
                viz_particles.row(i) = particles_.row(idx);
            }

            publish_particles(viz_particles);
        }
        else
        {
            publish_particles(particles_);
        }
    }
}

void ParticleFilter::publish_particles(const Eigen::MatrixXd &particles_to_pub)
{
    auto pa = utils::particles_to_pose_array(particles_to_pub);
    pa.header.stamp = this->get_clock()->now();
    pa.header.frame_id = "/map";
    particle_pub_->publish(pa);
}

// --------------------------------- UTILITY FUNCTIONS ---------------------------------
double ParticleFilter::quaternion_to_angle(const geometry_msgs::msg::Quaternion &q)
{
    return utils::quaternion_to_yaw(q);
}

geometry_msgs::msg::Quaternion ParticleFilter::angle_to_quaternion(double angle)
{
    return utils::yaw_to_quaternion(angle);
}

Eigen::Matrix2d ParticleFilter::rotation_matrix(double angle)
{
    return utils::rotation_matrix(angle);
}

Eigen::Vector2d ParticleFilter::transform_to_lidar_frame(const Eigen::Vector3d &base_pose)
{
    // Transform LiDAR offset from base_link to world coordinates
    double cos_theta = std::cos(base_pose[2]);
    double sin_theta = std::sin(base_pose[2]);
    double lidar_world_x = base_pose[0] + cos_theta * LIDAR_OFFSET_X - sin_theta * LIDAR_OFFSET_Y;
    double lidar_world_y = base_pose[1] + sin_theta * LIDAR_OFFSET_X + cos_theta * LIDAR_OFFSET_Y;
    return Eigen::Vector2d(lidar_world_x, lidar_world_y);
}

} // namespace particle_filter_cpp

// --------------------------------- PROGRAM ENTRY POINT ---------------------------------
// Main function for standalone executable
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<particle_filter_cpp::ParticleFilter>());
    rclcpp::shutdown();
    return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(particle_filter_cpp::ParticleFilter)

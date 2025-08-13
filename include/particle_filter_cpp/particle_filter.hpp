// ================================================================================================
// PARTICLE FILTER HEADER - Monte Carlo Localization (MCL) Class Definition
// ================================================================================================
// Features: Multinomial resampling, velocity motion model, beam sensor model, ray casting
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include <unordered_set>
#include <atomic>

namespace particle_filter_cpp
{

class ParticleFilter : public rclcpp::Node
{
  public:
    explicit ParticleFilter(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  private:
    // --------------------------------- CORE MCL ALGORITHM ---------------------------------
    void MCL(const Eigen::Vector3d &action, const std::vector<float> &observation);
    void motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action);
    void sensor_model(const Eigen::MatrixXd &proposal_dist, const std::vector<float> &obs,
                      std::vector<double> &weights);
    Eigen::Vector3d expected_pose();

    // --------------------------------- INITIALIZATION ---------------------------------
    void initialize_global();
    void initialize_particles_pose(const Eigen::Vector3d &pose);
    void precompute_sensor_model();

    // --------------------------------- ROS2 CALLBACKS ---------------------------------
    void lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg);
    void clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr msg);

    // --------------------------------- MAP MANAGEMENT ---------------------------------
    void get_omap();

    // --------------------------------- DYNAMIC MAP MANAGEMENT ---------------------------------
    void update_dynamic_obstacles(const Eigen::Vector3d& pose, 
                                 const std::vector<float>& ranges,
                                 const std::vector<float>& angles);
    void update_cell(int x, int y, int8_t new_value);
    void publish_dynamic_map_50hz();
    bool is_dynamic_obstacle(const Eigen::Vector3d& pose, float angle, float observed_range, float expected_range);

    // --------------------------------- OUTPUT & VISUALIZATION ---------------------------------
    void publish_tf(const Eigen::Vector3d &pose, const rclcpp::Time &stamp);
    void publish_odom_100hz();
    void visualize();
    void publish_particles(const Eigen::MatrixXd &particles_to_pub);
    
    // --------------------------------- POSE MANAGEMENT ---------------------------------
    Eigen::Vector3d get_current_pose();
    bool is_pose_valid(const Eigen::Vector3d& pose);

    // --------------------------------- UTILITY FUNCTIONS ---------------------------------
    double quaternion_to_angle(const geometry_msgs::msg::Quaternion &q);
    geometry_msgs::msg::Quaternion angle_to_quaternion(double angle);
    Eigen::Matrix2d rotation_matrix(double angle);

    // --------------------------------- RAY CASTING ---------------------------------
    std::vector<float> calc_range_many(const Eigen::MatrixXd &queries);
    float cast_ray(double x, double y, double angle);

    // --------------------------------- ALGORITHM PARAMETERS ---------------------------------
    int ANGLE_STEP;
    int MAX_PARTICLES;
    int MAX_VIZ_PARTICLES;
    double INV_SQUASH_FACTOR;
    double MAX_RANGE_METERS;
    int THETA_DISCRETIZATION;
    std::string WHICH_RM;
    int RANGELIB_VAR;
    bool SHOW_FINE_TIMING;
    bool PUBLISH_ODOM;
    bool DO_VIZ;
    double TIMER_FREQUENCY;

    // --------------------------------- SENSOR MODEL PARAMETERS ---------------------------------
    double Z_SHORT, Z_MAX, Z_RAND, Z_HIT, SIGMA_HIT;

    // --------------------------------- MOTION MODEL PARAMETERS ---------------------------------
    double MOTION_DISPERSION_X, MOTION_DISPERSION_Y, MOTION_DISPERSION_THETA;

    // --------------------------------- SENSOR FRAME PARAMETERS ---------------------------------
    double LIDAR_OFFSET_X, LIDAR_OFFSET_Y;
    double WHEELBASE;

    // --------------------------------- PARTICLE FILTER STATE ---------------------------------
    Eigen::MatrixXd particles_;
    std::vector<double> weights_;
    Eigen::Vector3d inferred_pose_;
    Eigen::Vector3d odometry_data_;
    Eigen::Vector3d last_pose_;

    // --------------------------------- SENSOR DATA ---------------------------------
    std::vector<float> laser_angles_;
    std::vector<float> downsampled_angles_;
    std::vector<float> downsampled_ranges_;

    // --------------------------------- MAP DATA ---------------------------------
    nav_msgs::msg::OccupancyGrid::SharedPtr map_msg_;
    Eigen::MatrixXi permissible_region_;
    bool map_initialized_;
    bool lidar_initialized_;
    bool odom_initialized_;
    bool first_sensor_update_;

    // --------------------------------- SENSOR MODEL OPTIMIZATION ---------------------------------
    Eigen::MatrixXd sensor_model_table_;
    int MAX_RANGE_PX;
    double map_resolution_;
    Eigen::Vector3d map_origin_;

    // --------------------------------- PERFORMANCE CACHES ---------------------------------
    Eigen::MatrixXd local_deltas_;
    Eigen::MatrixXd queries_;
    std::vector<float> ranges_;
    std::vector<float> tiled_angles_;

    // --------------------------------- ROS2 INTERFACES ---------------------------------
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_sub_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr dynamic_map_pub_;

    // Services and TF
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr map_client_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> pub_tf_;
    
    // Timer for high-frequency updates
    rclcpp::TimerBase::SharedPtr update_timer_;
    rclcpp::TimerBase::SharedPtr dynamic_map_timer_;

    // --------------------------------- THREADING ---------------------------------
    std::mutex state_lock_;

    // --------------------------------- RANDOM NUMBER GENERATION ---------------------------------
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;

    // --------------------------------- TIMING & STATISTICS ---------------------------------
    rclcpp::Time last_stamp_;
    int iters_;
    double current_speed_;
    double current_angular_velocity_;
    rclcpp::Time last_odom_time_;
    bool has_recent_odom_;
    
    // Interpolation state
    Eigen::Vector3d last_odom_motion_;
    rclcpp::Time last_odom_received_;
    rclcpp::Time prev_odom_received_;
    int steps_since_odom_;
    int expected_steps_between_odom_;
    Eigen::Vector3d accumulated_timer_motion_;

    // --------------------------------- ALGORITHM INTERNALS ---------------------------------
    std::vector<int> particle_indices_;

    // --------------------------------- DYNAMIC MAP STATE ---------------------------------
    std::vector<int8_t> dynamic_layer_;
    std::unordered_set<int> dirty_cells_;
    std::atomic<bool> has_changes_;
    nav_msgs::msg::OccupancyGrid::SharedPtr cached_dynamic_map_;
    double DYNAMIC_THRESHOLD_;
    double CLEAR_THRESHOLD_;
    int map_width_;
    int map_height_;

    // --------------------------------- UPDATE CONTROL ---------------------------------
    void update();
    void timer_update();
    void apply_interpolated_motion();
    bool should_interpolate_motion();
};

} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_

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
    
    // --------------------------------- SENSOR MODEL HELPERS ---------------------------------
    std::vector<float> convert_to_pixels(const std::vector<float> &ranges);
    void compute_particle_weights(const std::vector<float> &obs_px, const std::vector<float> &ranges_px,
                                std::vector<double> &weights, int num_rays);
    void normalize_weights();

    // --------------------------------- INITIALIZATION ---------------------------------
    void declare_parameters();
    void load_parameters();
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
    Eigen::Vector3d laser_to_base_link_pose(const Eigen::Vector3d &laser_pose);
    Eigen::Vector3d base_link_to_laser_pose(const Eigen::Vector3d &base_link_pose);

    // --------------------------------- RAY CASTING ---------------------------------
    std::vector<float> calc_range_many(const Eigen::MatrixXd &queries);
    float cast_ray(double x, double y, double angle);

    // --------------------------------- CONFIGURATION PARAMETERS ---------------------------------
    // Algorithm parameters
    int angle_step_;
    int max_particles_;
    int max_viz_particles_;
    double inv_squash_factor_;
    double max_range_meters_;
    int theta_discretization_;
    std::string range_method_;
    int rangelib_variant_;
    bool show_fine_timing_;
    bool publish_odom_;
    bool do_viz_;
    double timer_frequency_;
    bool use_parallel_raycasting_;
    int num_threads_;

    // Sensor model parameters (4-component beam model)
    double z_short_, z_max_, z_rand_, z_hit_, sigma_hit_;

    // Motion model noise parameters
    double motion_dispersion_x_, motion_dispersion_y_, motion_dispersion_theta_;

    // Robot geometry parameters
    double lidar_offset_x_, lidar_offset_y_;
    double wheelbase_;

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
    int max_range_px_;
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

    // Services and TF
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr map_client_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> pub_tf_;
    
    // Timer for high-frequency updates
    rclcpp::TimerBase::SharedPtr update_timer_;

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
    
    // Performance profiling
    struct TimingStats {
        double total_mcl_time = 0.0;
        double ray_casting_time = 0.0;
        double sensor_model_time = 0.0;
        double motion_model_time = 0.0;
        double resampling_time = 0.0;
        double query_prep_time = 0.0;
        int measurement_count = 0;
    } timing_stats_;
    

    // --------------------------------- ALGORITHM INTERNALS ---------------------------------
    std::vector<int> particle_indices_;

    // --------------------------------- UPDATE CONTROL ---------------------------------
    void update();
    void timer_update();
    
    // Performance profiling methods
    void print_performance_stats();
    void reset_performance_stats();
};

} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_

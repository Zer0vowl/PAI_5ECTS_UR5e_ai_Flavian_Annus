import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
import csv
import os
import sys
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

class EELogger(Node):
    def __init__(self, default_csv='ee_traj_dmp.csv'):
        super().__init__('ee_logger')

        # TF buffer + listener to read world -> tool0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # how often to sample (50 Hz to match your controller)
        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self.timer_cb)

        # where to store logs (change if you want)
        log_dir = os.path.expanduser('~/pai_ws/ee_logs')
        os.makedirs(log_dir, exist_ok=True)

        # configurable filename via parameter
        self.declare_parameter('csv_name', default_csv)
        csv_name = self.get_parameter('csv_name').get_parameter_value().string_value
        self.csv_path = os.path.join(log_dir, csv_name)

        self.get_logger().info(f'Logging EE trajectory to {self.csv_path}')

        # open CSV file
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        # header: time [s], x,y,z in world frame
        self.writer.writerow(['t', 'x', 'y', 'z'])

        # remember start time for relative timing
        self.t0 = self.get_clock().now()
        

        self.logging_enabled = False
        log_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool, "/ee_log_enable",
                                 lambda msg: setattr(self, "logging_enabled", msg.data),
                                 log_qos)

    def timer_cb(self):
        now = self.get_clock().now()
        try:
            if not self.logging_enabled:
                return   # skip rows until ramp is finished
            # world -> tool0 (change frame_ids if your setup differs)
            tf = self.tf_buffer.lookup_transform(
                'world', 'tool0', Time()
            )
        except tf2_ros.TransformException:
            # might be unavailable for the first few ticks
            return

        t = (now - self.t0).nanoseconds * 1e-9
        tr = tf.transform.translation
        x, y, z = tr.x, tr.y, tr.z

        self.writer.writerow([f'{t:.6f}', f'{x:.6f}', f'{y:.6f}', f'{z:.6f}'])

    def destroy_node(self):
        self.get_logger().info('Closing CSV file.')
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    # Simple flag parsing for convenience (-moveit or -dmp) to set default CSV name.
    default_csv = 'ee_traj_dmp.csv'
    for flag, name in [('-moveit', 'ee_traj_moveit.csv'), ('--moveit', 'ee_traj_moveit.csv'),
                       ('-dmp', 'ee_traj_dmp.csv'), ('--dmp', 'ee_traj_dmp.csv')]:
        if flag in sys.argv:
            default_csv = name
            # remove custom flag so rclpy doesn't choke on unknown args
            sys.argv.remove(flag)
    rclpy.init(args=sys.argv)
    node = EELogger(default_csv=default_csv)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

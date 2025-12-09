import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, ReliabilityPolicy
import tf2_ros

class EETrail(Node):
    def __init__(self):
        super().__init__('ee_trail_viz')
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(Marker, '/ee_trail', qos)

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)
        self.trail = []
        self.max_pts = 3000
        self.timer = self.create_timer(0.05, self.tick)   # 20 Hz

    def tick(self):
        try:
            t = self.buffer.lookup_transform('world', 'tool0', rclpy.time.Time())
            p = t.transform.translation
            self.trail.append(Point(x=p.x, y=p.y, z=p.z))
            if len(self.trail) > self.max_pts:
                self.trail = self.trail[-self.max_pts:]

            m = Marker()
            m.header.frame_id = 'world'
            m.ns = 'ee_trail'
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.005
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.1, 0.9, 1.0  # blue
            m.points = self.trail
            self.pub.publish(m)
        except Exception:
            pass

def main():
    rclpy.init()
    n = EETrail()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

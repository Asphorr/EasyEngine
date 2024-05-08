import pygame
import moderngl
import pygame_gui
from pygame.locals import *
from pygame_gui.elements import UIButton, UIDropDownMenu, UILabel, UIPanel
from pygame_gui import UIManager
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image
import numpy as np
import math

vertex_shader = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 frag_normal;
out vec3 frag_position;
out vec2 frag_texcoord;

void main() {
    frag_position = vec3(model * vec4(in_position, 1.0));
    frag_normal = mat3(transpose(inverse(model))) * in_normal;
    frag_texcoord = in_texcoord;

    gl_Position = projection * view * vec4(frag_position, 1.0);
}
"""

fragment_shader = """
#version 330
in vec3 frag_normal;
in vec3 frag_position;
in vec2 frag_texcoord;

uniform vec3 light_position;
uniform vec3 view_position;
uniform sampler2D texture_sampler;

out vec4 out_color;

void main() {
    // Ambient lighting
    float ambient_strength = 0.1;
    vec3 ambient = ambient_strength * vec3(1.0, 1.0, 1.0);

    // Diffuse lighting
    vec3 norm = normalize(frag_normal);
    vec3 light_dir = normalize(light_position - frag_position);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

    // Specular lighting
    float specular_strength = 0.5;
    vec3 view_dir = normalize(view_position - frag_position);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);

    // Combine results
    vec3 lighting = ambient + diffuse + specular;
    vec3 texture_color = texture(texture_sampler, frag_texcoord).rgb;

    out_color = vec4(lighting * texture_color, 1.0);
}
"""

class FreeCamera:
    def __init__(self, position, yaw=-90.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.cross(self.front, self.up)

        # Euler angles
        self.yaw = yaw
        self.pitch = pitch

        self.movement_speed = 5.0
        self.mouse_sensitivity = 0.1
        self.zoom = 45.0

        self.last_x = 400
        self.last_y = 300
        self.first_mouse = True

        self.update_camera_vectors()

    def update_camera_vectors(self):
        front = np.array(
            [
                math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
                math.sin(math.radians(self.pitch)),
                math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            ],
            dtype=np.float32,
        )
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, self.up)
        self.right = self.right / np.linalg.norm(self.right)

    def get_view_matrix(self):
        return self.look_at(self.position, self.position + self.front, self.up)

    @staticmethod
    def look_at(position, target, up):
        zaxis = position - target
        zaxis = zaxis / np.linalg.norm(zaxis)
        xaxis = np.cross(up, zaxis)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)

        translation = np.identity(4)
        translation[3, :3] = -position

        rotation = np.identity(4)
        rotation[:3, :3] = np.stack([xaxis, yaxis, zaxis], axis=0)

        return rotation @ translation

    def process_keyboard(self, direction, delta_time):
        velocity = self.movement_speed * delta_time
        if direction == "FORWARD":
            self.position += self.front * velocity
        if direction == "BACKWARD":
            self.position -= self.front * velocity
        if direction == "LEFT":
            self.position -= self.right * velocity
        if direction == "RIGHT":
            self.position += self.right * velocity

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self.update_camera_vectors()

class Sphere:
    def __init__(self, radius, lat_bands, long_bands):
        self.radius = radius
        self.lat_bands = lat_bands
        self.long_bands = long_bands

        vertices, normals, texcoords, indices = self.generate_sphere()
        self.vertex_data = np.array(
            [
                vertices[i] + normals[i] + texcoords[i]
                for i in range(len(vertices))
            ],
            dtype=np.float32,
        ).flatten()

        self.index_data = np.array(indices, dtype=np.uint32).flatten()

    def generate_sphere(self):
        vertices = []
        normals = []
        texcoords = []
        indices = []

        for lat in range(self.lat_bands + 1):
            theta = lat * math.pi / self.lat_bands
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            for lon in range(self.long_bands + 1):
                phi = lon * 2 * math.pi / self.long_bands
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)

                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta
                u = 1 - (lon / self.long_bands)
                v = 1 - (lat / self.lat_bands)

                vertices.append([self.radius * x, self.radius * y, self.radius * z])
                normals.append([x, y, z])
                texcoords.append([u, v])

        for lat in range(self.lat_bands):
            for lon in range(self.long_bands):
                first = (lat * (self.long_bands + 1)) + lon
                second = first + self.long_bands + 1

                indices.append(first)
                indices.append(second)
                indices.append(first + 1)

                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

        return vertices, normals, texcoords, indices

class Small3DEngine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Advanced 3D Engine")
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        self.screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
        self.ctx = moderngl.create_context()

        # Setup UI
        self.ui_manager = UIManager((800, 600))
        self.ui_elements = self.create_ui_elements()

        # OpenGL Initialization
        self.shader_program = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER),
        )

        self.sphere = Sphere(1.0, 30, 30)
        self.camera = FreeCamera([0.0, 0.0, 5.0])

        self.texture = self.load_texture("earth_texture.jpg")

        glUseProgram(self.shader_program)
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.sphere.vertex_data.nbytes, self.sphere.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.sphere.index_data.nbytes, self.sphere.index_data, GL_STATIC_DRAW)

        # Position attribute
        position = glGetAttribLocation(self.shader_program, "in_position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        # Normal attribute
        normal = glGetAttribLocation(self.shader_program, "in_normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(normal)

        # Texcoord attribute
        texcoord = glGetAttribLocation(self.shader_program, "in_texcoord")
        glVertexAttribPointer(texcoord, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        glEnableVertexAttribArray(texcoord)

        glBindVertexArray(0)

        # Uniform locations
        self.model_loc = glGetUniformLocation(self.shader_program, "model")
        self.view_loc = glGetUniformLocation(self.shader_program, "view")
        self.projection_loc = glGetUniformLocation(self.shader_program, "projection")
        self.light_pos_loc = glGetUniformLocation(self.shader_program, "light_position")
        self.view_pos_loc = glGetUniformLocation(self.shader_program, "view_position")
        self.texture_sampler_loc = glGetUniformLocation(self.shader_program, "texture_sampler")

        # Sphere model matrix
        self.model = np.identity(4, dtype=np.float32)

        # Light and view positions
        self.light_position = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        # Camera settings
        self.projection = np.identity(4, dtype=np.float32)

        # Set the active texture unit to the first one
        glUseProgram(self.shader_program)
        glUniform1i(self.texture_sampler_loc, 0)

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)

        # UI Settings
        self.environment = "Sunny"
        self.ray_tracing = False

    def create_ui_elements(self):
        panel = UIPanel(
            relative_rect=pygame.Rect((10, 10), (300, 150)),
            manager=self.ui_manager,
            object_id="#main_panel"
        )

        self.env_label = UILabel(
            relative_rect=pygame.Rect((10, 10), (80, 30)),
            text="Environment:",
            manager=self.ui_manager,
            container=panel
        )

        self.environment_ddm = UIDropDownMenu(
            options_list=["Sunny", "Cloudy", "Rain"],
            starting_option="Sunny",
            relative_rect=pygame.Rect((100, 10), (150, 30)),
            manager=self.ui_manager,
            container=panel
        )

        self.toggle_ray_tracing = UIButton(
            relative_rect=pygame.Rect((10, 50), (150, 30)),
            text="Toggle Ray Tracing",
            manager=self.ui_manager,
            container=panel
        )

        return {
            "panel": panel,
            "environment_ddm": self.environment_ddm,
            "toggle_ray_tracing": self.toggle_ray_tracing
        }

    def load_texture(self, path):
        image = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
        image_data = image.convert("RGB").tobytes()
        texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        return texture

    def handle_events(self, event):
        self.ui_manager.process_events(event)

        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.ui_elements["toggle_ray_tracing"]:
                    self.ray_tracing = not self.ray_tracing

            elif event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == self.ui_elements["environment_ddm"]:
                    self.environment = event.text

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit(0)

    def update_camera(self, delta_time):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            self.camera.process_keyboard("FORWARD", delta_time)
        if keys[pygame.K_s]:
            self.camera.process_keyboard("BACKWARD", delta_time)
        if keys[pygame.K_a]:
            self.camera.process_keyboard("LEFT", delta_time)
        if keys[pygame.K_d]:
            self.camera.process_keyboard("RIGHT", delta_time)

        x, y = pygame.mouse.get_pos()

        if self.camera.first_mouse:
            self.camera.last_x = x
            self.camera.last_y = y
            self.camera.first_mouse = False

        x_offset = x - self.camera.last_x
        y_offset = self.camera.last_y - y

        self.camera.last_x = x
        self.camera.last_y = y

        self.camera.process_mouse_movement(x_offset, y_offset)

    def apply_environment(self):
        if self.environment == "Sunny":
            glClearColor(0.53, 0.81, 0.98, 1)
        elif self.environment == "Cloudy":
            glClearColor(0.7, 0.7, 0.7, 1)
        elif self.environment == "Rain":
            glClearColor(0.3, 0.3, 0.5, 1)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = self.camera.get_view_matrix()
        projection = np.identity(4, dtype=np.float32)
        fov = math.radians(self.camera.zoom)
        aspect_ratio = 800 / 600
        near = 0.1
        far = 100.0

        projection[0, 0] = 1 / (aspect_ratio * math.tan(fov / 2))
        projection[1, 1] = 1 / math.tan(fov / 2)
        projection[2, 2] = -(far + near) / (far - near)
        projection[2, 3] = -1
        projection[3, 2] = -(2 * far * near) / (far - near)

        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.model)
        glUniform3fv(self.light_pos_loc, 1, self.light_position)
        glUniform3fv(self.view_pos_loc, 1, self.camera.position)

        glBindVertexArray(self.vao)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawElements(GL_TRIANGLES, len(self.sphere.index_data), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            time_delta = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                else:
                    self.handle_events(event)

            self.apply_environment()
            self.update_camera(time_delta)
            self.ui_manager.update(time_delta)

            self.render()
            self.ui_manager.draw_ui(self.screen)

        pygame.quit()


if __name__ == "__main__":
    engine = Small3DEngine()
    engine.run()

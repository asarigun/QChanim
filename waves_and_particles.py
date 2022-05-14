import numpy as np
import statistics
import random
from manimlib import *


COLOR = 'red'

"""
References:

This implementation would not make out without the codes that I used as reference.

-https://github.com/3b1b/manim
-https://github.com/nipunramk/Reducible
-https://pastebin.com/fS5vAM5r

"""

class Introduction(Scene):

      def construct(self):
        self.intro()
        self.wave()
        self.wavelength()

      def intro(self):
        title = TextMobject("Waves and Particles")
        title.scale(1.2)
        title.shift(UP * 3.5)

        self.h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        self.h_line.next_to(title, DOWN)
        self.play(
                Write(title),
                ShowCreation(self.h_line)
            )
        self.wait()

        classical_phys = TextMobject("According to the classical physics \\\\ (ie. 100 years ago)")
        classical_phys.scale(0.8)
        classical_phys.next_to(self.h_line, DOWN)
        self.play(
            Write(classical_phys)
        )
        self.wait()
        
        conference= ImageMobject("assets/Solvayconference1911.jpg")
        conference.scale(2)
        conference.next_to(classical_phys, DOWN)
        self.add(conference)
        self.wait(1)
        solvay_txt = TextMobject("Solvay Conference, 1911")
        solvay_txt.scale(0.8)
        solvay_txt.next_to(conference, DOWN)
        self.play(
            Write(solvay_txt)
        )
                
        self.wait(2)
        self.play(
                FadeOut(title),
                FadeOut(self.h_line),
                FadeOut(classical_phys),
                FadeOut(conference),
                FadeOut(solvay_txt),
            )
      def wave(self):
        title = TextMobject("Waves")
        title.set_color(YELLOW)
        title.scale(1.2)
        title.shift(UP * 3.5)
        self.h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        self.h_line.next_to(title, DOWN)

        self.play(
                Write(title),
                ShowCreation(self.h_line)
        )
        self.wait()
        text=TextMobject(r"Interference of waves both constructive and destructive and related" + "\\\\",
          r" with the interference phenomena is diffraction from surfaces of crystals"
        )
        text.scale(0.8)
        text.next_to(self.h_line, DOWN)
        for i in range(len(text)):
            self.play(
            Write(text[i]),
            run_time=3
          )

        # Background color
        self.camera.background_color = BLACK

        # Graph settings
        graph = Axes(
            
            x_length=13,
            y_length=7,
            axis_config={
                'color' : WHITE,
                'stroke_width' : 4,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : True,
                    'color' : WHITE
                }
            },

        )

        # Functions
        curve_1 = graph.get_graph(lambda x: 2*np.sin(x),color = WHITE)
        curve_2 = graph.get_graph(lambda x: np.sin(x),color = BLUE)
        curve_3 = graph.get_graph(lambda x: np.sin(x),color = RED)
        curve_1.scale(0.5)
        curve_2.scale(0.5)
        curve_3.scale(0.5)
        graph.scale(0.5)
        graph.next_to(text, DOWN)

        dash1 = DashedVMobject(curve_2)#.shift(3*LEFT)
        dash2 = DashedVMobject(curve_3)#.shift(3*LEFT)

        delta = TexMobject("\\Delta", "\\phi", "=" ,color = WHITE)
        cont = TexMobject("0.0", color = WHITE)
        p = TextMobject("°",color = WHITE)
        delta.next_to(graph, DOWN)

        cont.next_to(delta, 0.5*RIGHT)
        p.next_to(cont,0.05*RIGHT + 0.01*UP)

        n = 100
        rate = 0
        rate2 = PI/100
        self.add(graph, curve_1, dash1,dash2,cont,delta,p)
        for i in range(n+1):
            rate += rate2
            s = rate*180/PI
 
            func_graph11 = graph.get_graph(lambda x: 2*np.cos(rate/2)*np.sin(x  - rate/2) , color =WHITE)
            func_graph21 = graph.get_graph(lambda x: np.sin(x - rate), color = RED)
            dash21 = DashedVMobject(func_graph21)#.shift(3*LEFT)
 
            contador = DecimalNumber(s , num_decimal_places= 1, color = WHITE)
            p2 = TextMobject("°",color = WHITE)
 
 
            contador.next_to(delta, 0.5*RIGHT)
            p2.next_to(contador,0.05*RIGHT+ 0.01*UP)
 
            self.play(Transform(curve_1,func_graph11),Transform(dash2,dash21), 
                Transform(cont,contador),Transform(p,p2), run_time = 0.0001)
        self.wait(3)
        self.play(
                FadeOut(text),
                FadeOut(curve_1),
                FadeOut(dash1),
                FadeOut(dash2),
                FadeOut(cont),
                FadeOut(delta),
                FadeOut(p),
                FadeOut(graph),
            )
        
      def wavelength(self):
        graph = Axes(
            #x_range=np.array([-8, 8, 2]),
            #y_range=np.array([-4, 4, 2]),
            x_length=13,
            y_length=7,
            axis_config={
                'color' : WHITE,
                'stroke_width' : 4,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : True,
                    'color' : WHITE
                }
            },

        )
        title = TextMobject("Waves")
        title.set_color(YELLOW)
        title.scale(1.2)
        title.shift(UP * 3.5)
        self.h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        self.h_line.next_to(title, DOWN)

        self.play(
                Write(title),
                ShowCreation(self.h_line)
        )
        self.wait()
        wave = graph.get_graph(lambda x: np.sin(x),color = BLUE)#.get_points()
        wave.scale(0.5)
        point1 = wave.get_point_from_function(0)
        point2 = wave.get_point_from_function((2*np.pi))
        d_arrow = DoubleArrow(start=point1, end=point2)
        d_arrow.scale(0.55)
        d_arrow.shift((1.25*np.pi)*LEFT, 1*UP)
        #b1 = BraceBetweenPoint(point1,point2)
        lamb = TexMobject("\\lambda")
        lamb.next_to(d_arrow, UP)
        #lamb.scale(0.8)
        self.play(ShowCreation(wave), ShowCreation(d_arrow), Write(lamb))#, run_time=2.0)

class Ball(Circle):
    CONFIG = {
        "radius": 0.15,
        "fill_color": BLUE,
        "fill_opacity": 1,
        "color": BLUE,
        "id": 0, 
    }

    def __init__(self, **kwargs):
        Circle.__init__(self, ** kwargs)
        self.velocity = np.array((2, 0, 0))
        self.mass = PI * self.radius ** 2

    ### Only gives scalar values ###
    def get_top(self):
        return self.get_center()[1] + self.radius

    def get_bottom(self):
        return self.get_center()[1] - self.radius

    def get_right_edge(self):
        return self.get_center()[0] + self.radius

    def get_left_edge(self):
        return self.get_center()[0] - self.radius

    ### Gives full vector ###
    def get_top_v(self):
        return self.get_center() + self.radius * UP

    def get_bottom_v(self):
        return self.get_center() + self.radius * DOWN

    def get_right_edge_v(self):
        return self.get_center() + self.radius * RIGHT

    def get_left_edge_v(self):
        return self.get_center() + self.radius * LEFT

class Box(Rectangle):
    CONFIG = {
        "height": 6,
        "width": FRAME_WIDTH - 1,
        "color": GREEN_C
    }

    def __init__(self, **kwargs):
        Rectangle.__init__(self, ** kwargs) 

    def get_top(self):
        return self.get_center()[1] + (self.height / 2)

    def get_bottom(self):
        return self.get_center()[1] - (self.height / 2)

    def get_right_edge(self):
        return self.get_center()[0] + (self.width / 2)

    def get_left_edge(self):
        return self.get_center()[0] - (self.width / 2)




class IntroHandlingSeveralParticles(Scene):
    CONFIG = {
        "simulation_time": 30,
    }
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 6
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(YELLOW)
        shift_right = RIGHT * 4
        box.shift(shift_right)
        velocities = [
        LEFT * 1 + UP * 1, RIGHT * 1, LEFT + DOWN * 1, 
        RIGHT + DOWN * 1, RIGHT * 0.5 + DOWN * 0.5, RIGHT * 0.5 + UP * 0.5
        ]
        positions = [
        LEFT * 2 + UP * 1, UP * 1, RIGHT * 2 + UP * 1,
        LEFT * 2 + DOWN * 1, DOWN * 1, RIGHT * 2 + DOWN * 1,
        ]
        colors = [WHITE, MAROON, TEAL, GOLD, LIGHT_BROWN, PURPLE_A]
        radius =[0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box)
        )

        def update_particles(particles, dt):
            for i in range(len(particles)):
                particle = particles[i]
                particle.acceleration = np.array((0, 0, 0))
                particle.velocity = particle.velocity + particle.acceleration * dt
                particle.shift(particle.velocity * dt)
                handle_collision_with_box(particle, box, dt)
            
            handle_particle_collisions_opt(particles, dt)

        def handle_collision_with_box(particle, box, dt):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                particle.velocity[1] = -particle.velocity[1]
                particle.shift(particle.velocity * dt)
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() * BOX_THRESHOLD or \
                    particle.get_right_edge() >= box.get_right_edge() * BOX_THRESHOLD :
                particle.velocity[0] = -particle.velocity[0]
                particle.shift(particle.velocity * dt)

        def handle_particle_collisions(particles):
            for particle in particles:
                for other_particle in particles:
                    if particle.id != other_particle.id:
                        dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                        if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                            # ball.set_color(random.choice(t_colors))
                            # other_ball.set_color(random.choice(t_colors))
                            v1, v2 = get_response_velocities(particle, other_particle)
                            particle.velocity = v1
                            other_particle.velocity = v2

        def handle_particle_collisions_opt(particles, dt):
            possible_collisions = find_possible_collisions(particles) 
            # print([(a.id, b.id) for a, b in possible_collisions])
            # print(len(possible_collisions))
            for particle, other_particle in possible_collisions:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
                        particle.shift(particle.velocity * dt)
                        other_particle.shift(other_particle.velocity * dt)
        
        def find_possible_collisions(particles):
            # implements the sort and sweep algorithm for broad phase
            # helpful reference: https://github.com/mattleibow/jitterphysics/wiki/Sweep-and-Prune
            axis_list = sorted(particles, key=lambda x: x.get_left()[0])
            active_list = []
            possible_collisions = set()
            for particle in axis_list:
                to_remove = [p for p in active_list if particle.get_left()[0] > p.get_right()[0]]
                for r in to_remove:
                    active_list.remove(r)
                for other_particle in active_list:
                    possible_collisions.add((particle, other_particle))

                active_list.append(particle)
            
            return possible_collisions

        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        def show_all_collisions(particles):
            transforms = []
            end_positions = []
            all_pairs = []
            rows = [UP * 2, UP * 1, UP * 0, DOWN * 1, DOWN * 2]
            columns = [LEFT * 5.5, LEFT * 4, LEFT * 2.5]
            for col in columns:
                for row in rows:
                    end_positions.append(row + col)
            seen = set()
            i = 0
            for p1 in particles:
                for p2 in particles:
                    if p1.id == p2.id or (p2.id, p1.id) in seen:
                        continue
                    pair = VGroup(p1, p2)
                    p1_c = p1.copy().scale(0.7)
                    p2_c = p2.copy().scale(0.7)

                    p2_c.next_to(p1_c, RIGHT, buff=SMALL_BUFF)
                    transform_pair = VGroup(p1_c, p2_c).move_to(end_positions[i])
                    all_pairs.append(transform_pair)

                    transforms.append(TransformFromCopy(pair, transform_pair))
                    i += 1
                    seen.add((p1.id, p2.id))

            return transforms, all_pairs

        particles = VGroup(*particles)
        self.play(
            FadeIn(particles)
        )
        particles.add_updater(update_particles)
        self.add(particles)
        self.wait(16)
        particles.clear_updaters()

        print("--- %s seconds ---" % (time.time() - start_time))
        
        
        #self.wait(3)

        idea = TextMobject("Properties of particles").scale(1)
        idea.move_to(UP * 3).to_edge(LEFT * 2)
        h_line = Line(LEFT * 7.5, LEFT * 2.0)
        indent = LEFT * 1.5
        h_line.next_to(idea, DOWN).to_edge(indent)

        
        self.play(
            Write(idea),
            ShowCreation(h_line)
        )

        self.wait()

        topics = BulletedList(
            "Mass",
            "Well-defined Position",
            "Linear Momentum",
            "Countable",
            ).scale(0.9)

        topics.next_to(idea, DOWN).shift(DOWN * 1.0)
        for i in range(len(topics)):
            self.play(
                Write(topics[i])
            )
            self.wait(3)

        self.play(
            *[FadeOut(p) for p in particles],
        )
        
class TwoParticleSim(Scene):
    CONFIG = {
        "sim_time": 45,
    }
    def construct(self):
        debug = False
        balls = []
        num_balls = 2
        BOX_THRESHOLD = 0.98
        BALL_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(YELLOW)
        box.shift(RIGHT * 4)
        colors = [WHITE, RED_C, GREEN_SCREEN, ORANGE]
        velocities = [RIGHT * 2 + UP * 2, LEFT * 1 + UP * 2]
        positions = [RIGHT * 3, RIGHT * 5]
        
        for i in range(len(positions)):
            if i == 0:
                ball = Ball(
                    radius=0.3, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            else:
                ball = Ball(
                    radius=0.4, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            ball.id = i
            ball.move_to(positions[i])
            ball.velocity = velocities[i]
            balls.append(ball)
        
        self.play(
            FadeIn(box)
        )
        self.play(
            *[FadeIn(ball) for ball in balls]
        )

        # useful info for debugging
        p1_value_x = round(balls[0].get_center()[0], 3)
        p1_value_y = round(balls[0].get_center()[1], 3)
        p1_text = TextMobject("Position: ")
        p1 = VGroup(p1_text, DecimalNumber(p1_value_x), DecimalNumber(p1_value_y))
        p1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        v1_value_x = round(balls[0].velocity[0], 3)
        v1_value_y = round(balls[0].velocity[1], 3)
        v1_text = TextMobject("Velocity: ")
        v1 = VGroup(v1_text, DecimalNumber(v1_value_x), DecimalNumber(v1_value_y))
        v1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        p2_value_x = round(balls[1].get_center()[0], 3)
        p2_value_y = round(balls[1].get_center()[1], 3)
        p2_text = TextMobject("Position: ")
        p2 = VGroup(p2_text, DecimalNumber(p2_value_x), DecimalNumber(p2_value_y))
        p2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        v2_value_x = round(balls[1].velocity[0], 3)
        v2_value_y = round(balls[1].velocity[1], 3)
        v2_text = TextMobject("Velocity: ")
        v2 = VGroup(v2_text, DecimalNumber(v2_value_x), DecimalNumber(v2_value_y))
        v2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        if debug:
            debug_log = VGroup(p1, v1, p2, v2).arrange_submobjects(DOWN) 
            debug_log.shift(LEFT * 4)
            self.play(
                FadeIn(debug_log)
            )



        def update_ball(ball, dt):
            ball.acceleration = np.array((0, 0, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)
            handle_collision_with_box(ball, box, dt)
            handle_ball_collisions(ball, dt)

            if ball.get_color() == Color(BLUE) and debug:
                p1[1].set_value(ball.get_center()[0])
                p1[2].set_value(ball.get_center()[1])
                v1[1].set_value(ball.velocity[0])
                v1[2].set_value(ball.velocity[1])

            if ball.get_color() == Color(YELLOW) and debug:
                p2[1].set_value(ball.get_center()[0])
                p2[2].set_value(ball.get_center()[1])
                v2[1].set_value(ball.velocity[0])
                v2[2].set_value(ball.velocity[1])

        def handle_collision_with_box(ball, box, dt):
            # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top()*BOX_THRESHOLD:
                ball.velocity[1] = -ball.velocity[1]
                ball.shift(ball.velocity * dt)
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() or \
                    ball.get_right_edge() >= box.get_right_edge():
                ball.velocity[0] = -ball.velocity[0]
                ball.shift(ball.velocity * dt)

        def handle_ball_collisions(ball, dt):
            t_colors = [RED, ORANGE, GREEN_SCREEN, GOLD, PINK, WHITE]
            for other_ball in balls:
                if ball.id != other_ball.id:
                    dist = np.linalg.norm(ball.get_center() - other_ball.get_center())
                    if dist * BALL_THRESHOLD <= (ball.radius + other_ball.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(ball, other_ball)
                        ball.velocity = v1
                        other_ball.velocity = v2
                        ball.shift(ball.velocity * dt)
                        other_ball.shift(other_ball.velocity * dt)
        
        def get_response_velocities(ball, other_ball):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = ball.velocity
            v2 = other_ball.velocity
            m1 = ball.mass
            m2 = other_ball.mass
            x1 = ball.get_center()
            x2 = other_ball.get_center()

            ball_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_ball_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return ball_response_v, other_ball_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        idea = TextMobject("Lineer Momentum").scale(1)
        idea.move_to(UP * 3).to_edge(LEFT * 2)
        h_line = Line(LEFT * 7.5, LEFT * 2.0)
        indent = LEFT * 1.5
        h_line.next_to(idea, DOWN).to_edge(indent)

        
        self.play(
            Write(idea),
            ShowCreation(h_line)
        )

        self.wait()

        topics = BulletedList(
            "Related mass(m) and velocity($v$)",
            "Conservation of Momentum"
            ).scale(0.8)

        topics[0].next_to(idea, DOWN).shift(DOWN * 1.0, RIGHT * 0.80)
        self.play(
            Write(topics[0])
            )
        self.wait()
        momentum = TexMobject(r"p = mv")
        momentum.scale(0.9)
        momentum.next_to(topics[0], DOWN)#.shift(DOWN * 1.0)
        self.play(
            Write(momentum)
            )
        self.wait()

        topics[1].next_to(topics[0], DOWN).shift(DOWN * 1.0, LEFT * 0.33)
        self.play(
            Write(topics[1])
            )
        self.wait()
        momentum2 = TexMobject(r"p_{total} = p_1 + p_2")
        momentum2.scale(0.9)
        momentum2.next_to(topics[1], DOWN)#.shift(DOWN * 1.0)
        self.play(
            Write(momentum2)
            )

        momentum3 = TexMobject(r"p_1 + p_2 = p_1^{\prime} + p_2^{\prime}")
        momentum3.scale(0.9)
        momentum3.next_to(momentum2, DOWN)#.shift(DOWN * 1.0)
        self.play(
            Write(momentum3)
            )
        
        for ball in balls:
            ball.add_updater(update_ball)
            self.add(ball)

        self.wait(self.sim_time)
        for ball in balls:
            ball.clear_updaters()
        self.wait()

class ParticleSimulation(Scene):
    CONFIG = {
        "simulation_time": 10,
    }
    def construct(self):
        start_time = time.time()
        particles = []
        #particles.shift(LEFT*4)
        num_particles = 100
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5)
        box.shift(LEFT * 4)

        graph = Axes(
            #x_range=np.array([-8, 8, 2]),
            #y_range=np.array([-4, 4, 2]),
            x_length=13,
            y_length=7,
            axis_config={
                'color' : WHITE,
                'stroke_width' : 4,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : True,
                    'color' : WHITE
                }
            },

        )
        

        
        self.wait()
        wave = graph.get_graph(lambda x: np.sin(x),color = BLUE)#.get_points()
        wave.scale(0.5)
        wave.shift(RIGHT*3)

        point1 = box
        point2 = wave
        d_arrow = DoubleArrow(start=point1, end=point2)
        d_arrow.scale(0.55)
        partic=TextMobject("Particles")
        partic.scale(0.8)
        partic.next_to(box, UP)

        wav=TextMobject("Waves")
        wav.scale(0.8)
        wav.next_to(wave, UP*5)

        velocities = [UP * i + RIGHT * j for i, j in zip(np.random.uniform(-0.2, 0.2, num_particles), np.random.uniform(-0.2, 0.2, num_particles))]
        positions = []
        for i in np.arange(-2.5, 2.5, 0.5):
            for j in np.arange(-2.5, 2.5, 0.5):
                positions.append(RIGHT * i + UP * j)
        for i in range(len(positions)):
            particle = Ball(radius=0.04)
            particle.shift(LEFT*4)
            particle.id = i
            particle.move_to(positions[i])
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box),
            Write(partic),
            #ShowCreation(d_arrow),
            ShowCreation(wave),
            Write(wav)
        )
        self.play(
            *[FadeIn(particle) for particle in particles]
        )

        def update_particle(particle, dt):
            particle.acceleration = np.array((0, 0, 0))
            particle.velocity = particle.velocity + particle.acceleration * dt
            particle.shift(particle.velocity * dt)
            handle_collision_with_box(particle, box)
            handle_particle_collisions(particle)

        def handle_collision_with_box(particle, box):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                    particle.velocity[1] = -particle.velocity[1]
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() or \
                    particle.get_right_edge() >= box.get_right_edge():
                particle.velocity[0] = -particle.velocity[0]

        def handle_particle_collisions(particle):
            t_colors = [RED, ORANGE, GREEN_SCREEN, GOLD, PINK, WHITE]
            i = 0
            for other_particle in particles:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
        
        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        for particle in particles:
            particle.add_updater(update_particle)
            self.add(particle)

        self.wait(self.simulation_time)
        for particle in particles:
            particle.clear_updaters()
        self.wait(3)
        print("--- %s seconds ---" % (time.time() - start_time))

class wavesParticles(Scene):
    CONFIG = {
        "simulation_time": 30,
    }
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 6
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(YELLOW)
        shift_right = RIGHT * 4
        box.shift(shift_right)
        velocities = [
        LEFT * 1 + UP * 1, RIGHT * 1, LEFT + DOWN * 1, 
        RIGHT + DOWN * 1, RIGHT * 0.5 + DOWN * 0.5, RIGHT * 0.5 + UP * 0.5
        ]
        positions = [
        LEFT * 2 + UP * 1, UP * 1, RIGHT * 2 + UP * 1,
        LEFT * 2 + DOWN * 1, DOWN * 1, RIGHT * 2 + DOWN * 1,
        ]
        colors = [WHITE, MAROON, TEAL, GOLD, LIGHT_BROWN, PURPLE_A]
        radius =[0.18, 0.19, 0.20, 0.21, 0.22, 0.23]

        graph = Axes(
            
            x_length=13,
            y_length=7,
            axis_config={
                'color' : WHITE,
                'stroke_width' : 4,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : True,
                    'color' : WHITE
                }
            },

        )
        

        
        self.wait()
        wave = graph.get_graph(lambda x: np.sin(x),color = BLUE)
        wave.scale(0.5)
        wave.shift(LEFT*3)

        point1 = box
        point2 = wave
        d_arrow = DoubleArrow(start=point1, end=point2)
        d_arrow.scale(0.55)
        partic=TextMobject("Particles")
        partic.scale(0.8)
        partic.next_to(box, UP)

        wav=TextMobject("Waves")
        wav.scale(0.8)
        wav.next_to(wave, UP*10)

        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box),
            Write(partic),
            ShowCreation(wave),
            Write(wav)
        )

        def update_particles(particles, dt):
            for i in range(len(particles)):
                particle = particles[i]
                particle.acceleration = np.array((0, 0, 0))
                particle.velocity = particle.velocity + particle.acceleration * dt
                particle.shift(particle.velocity * dt)
                handle_collision_with_box(particle, box, dt)
            
            handle_particle_collisions_opt(particles, dt)

        def handle_collision_with_box(particle, box, dt):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                particle.velocity[1] = -particle.velocity[1]
                particle.shift(particle.velocity * dt)
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() * BOX_THRESHOLD or \
                    particle.get_right_edge() >= box.get_right_edge() * BOX_THRESHOLD :
                particle.velocity[0] = -particle.velocity[0]
                particle.shift(particle.velocity * dt)

        def handle_particle_collisions(particles):
            for particle in particles:
                for other_particle in particles:
                    if particle.id != other_particle.id:
                        dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                        if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                            # ball.set_color(random.choice(t_colors))
                            # other_ball.set_color(random.choice(t_colors))
                            v1, v2 = get_response_velocities(particle, other_particle)
                            particle.velocity = v1
                            other_particle.velocity = v2

        def handle_particle_collisions_opt(particles, dt):
            possible_collisions = find_possible_collisions(particles) 
            # print([(a.id, b.id) for a, b in possible_collisions])
            # print(len(possible_collisions))
            for particle, other_particle in possible_collisions:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
                        particle.shift(particle.velocity * dt)
                        other_particle.shift(other_particle.velocity * dt)
        
        def find_possible_collisions(particles):
            # implements the sort and sweep algorithm for broad phase
            # helpful reference: https://github.com/mattleibow/jitterphysics/wiki/Sweep-and-Prune
            axis_list = sorted(particles, key=lambda x: x.get_left()[0])
            active_list = []
            possible_collisions = set()
            for particle in axis_list:
                to_remove = [p for p in active_list if particle.get_left()[0] > p.get_right()[0]]
                for r in to_remove:
                    active_list.remove(r)
                for other_particle in active_list:
                    possible_collisions.add((particle, other_particle))

                active_list.append(particle)
            
            return possible_collisions

        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        def show_all_collisions(particles):
            transforms = []
            end_positions = []
            all_pairs = []
            rows = [UP * 2, UP * 1, UP * 0, DOWN * 1, DOWN * 2]
            columns = [LEFT * 5.5, LEFT * 4, LEFT * 2.5]
            for col in columns:
                for row in rows:
                    end_positions.append(row + col)
            seen = set()
            i = 0
            for p1 in particles:
                for p2 in particles:
                    if p1.id == p2.id or (p2.id, p1.id) in seen:
                        continue
                    pair = VGroup(p1, p2)
                    p1_c = p1.copy().scale(0.7)
                    p2_c = p2.copy().scale(0.7)

                    p2_c.next_to(p1_c, RIGHT, buff=SMALL_BUFF)
                    transform_pair = VGroup(p1_c, p2_c).move_to(end_positions[i])
                    all_pairs.append(transform_pair)

                    transforms.append(TransformFromCopy(pair, transform_pair))
                    i += 1
                    seen.add((p1.id, p2.id))

            return transforms, all_pairs

        particles = VGroup(*particles)
        self.play(
            FadeIn(particles)
        )
        particles.add_updater(update_particles)
        self.add(particles)
        self.wait(16)
        particles.clear_updaters()

        print("--- %s seconds ---" % (time.time() - start_time))

class Light(Scene):

      def construct(self):
        #self.intro()
        self.wavelength()

      def intro(self):
        title = TextMobject("Light")
        title.scale(1.2)
        title.shift(UP * 3.5)

        self.h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        self.h_line.next_to(title, DOWN)
        self.play(
                Write(title),
                ShowCreation(self.h_line)
            )
        self.wait()

        classical_phys = TextMobject("Light is a form of energy that behaves like a wave.")
        classical_phys.scale(0.8)
        classical_phys.next_to(self.h_line, DOWN)
        self.play(
            Write(classical_phys)
        )
        self.wait()

      def wavelength(self):
        graph = Axes(
            x_length=13,
            y_length=7,
            axis_config={
                'color' : WHITE,
                'stroke_width' : 4,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : True,
                    'color' : WHITE
                }
            },

        )
        title = TextMobject("Light")
        title.set_color(YELLOW)
        title.scale(1.2)
        title.shift(UP * 3.5)
        self.h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        self.h_line.next_to(title, DOWN)

        self.play(
                Write(title),
                ShowCreation(self.h_line)
        )
        self.wait()
        classical_phys = TextMobject("Light is a form of energy that behaves like a wave.")
        classical_phys.scale(0.8)
        classical_phys.next_to(self.h_line, DOWN)
        self.play(
            Write(classical_phys)
        )
        self.wait()

        wave = graph.get_graph(lambda x: np.sin(x),color = BLUE)#.get_points()
        wave.next_to(classical_phys, DOWN * 5)
        wave.scale(0.5)
        point1 = wave.get_point_from_function(0)
        point2 = wave.get_point_from_function((2*np.pi))
        d_arrow = DoubleArrow(start=point1, end=point2)
        d_arrow.scale(0.55)
        d_arrow.shift((1.25*np.pi)*LEFT, 1*UP)
        lamb = TexMobject("\\lambda")
        lamb.next_to(d_arrow, UP)
        self.play(
            ShowCreation(wave), 
            ShowCreation(d_arrow), 
            Write(lamb)
            )
        self.wait()
        range = TextMobject("400 nm - 700 nm")
        range.next_to(wave, DOWN*3)
        range.scale(0.8)
        self.play(
            Write(range)
            )
        self.wait()
        
        br = Brace(mobject=range, direction=DOWN, buff=0.2)
        br_vis = br.get_text("Visible Range")
        br_vis.scale(0.8)
        br_vis.set_color(RED_A)
        self.play(
            GrowFromCenter(br), 
            FadeIn(br_vis), 
            run_time=2)
        self.wait()

        self.play(
                FadeOut(classical_phys),
                FadeOut(wave),
                FadeOut(d_arrow),
                FadeOut(lamb),
                FadeOut(range),
                FadeOut(br),
                FadeOut(br_vis)
            )
        self.wait()     
        speed=TextMobject("The speed of light in vacuum in empty space is the same for all wavelengths")
        speed.next_to(self.h_line, DOWN)
        speed.scale(0.8)
        self.play(
            Write(speed)
            )
        self.wait()
        c = TexMobject(r"c=3x10^{8} m/s")
        c.next_to(speed, DOWN)
        c.scale(0.8)
        self.play(
            Write(c)
            )
        self.wait()
        freq=TexMobject(r"\nu \lambda = same")
        freq.scale(0.8)
        freq.next_to(c,DOWN)
        self.play(
            Write(freq)
            )
        self.wait()
        self.play(
            FadeOut(freq)
            )
        freq_c = TexMobject(r"\nu \lambda = c = 3x10^{8} m/s")
        freq_c.next_to(c,DOWN)
        freq_c.scale(0.8)
        self.play(
            Write(freq_c)
            )
        self.wait()
        topics = BulletedList(
            "Wavelength: meter",
            "Frequency: 1/second",
            "Speed: meter/second",
            ).scale(0.9)

        topics[0].next_to(freq_c, DOWN*3)
        topics[0].set_color(GREEN_C)
        self.play(
            Write(topics[0])
            )
        self.wait()
        topics[1].next_to(topics[0], DOWN)
        topics[1].set_color(GREEN_C)
        self.play(
            Write(topics[1])
            )
        self.wait()
        topics[2].next_to(topics[1], DOWN)
        topics[2].set_color(GREEN_C)
        self.play(
            Write(topics[2])
            )
        self.wait()

        self.play(
            FadeOut(speed),
            FadeOut(c),
            FadeOut(freq_c),
            FadeOut(topics[0]),
            FadeOut(topics[1]),
            FadeOut(topics[2])
        )
        debug = False
        balls = []
        num_balls = 5
        BOX_THRESHOLD = 0.98
        BALL_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(BLUE)
        
        colors = [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE]
        
        positions = [RIGHT * 3, RIGHT * 5]
        
        
def create_computer_char(color=BLUE, scale=1, position=ORIGIN):
	outer_rectangle = Rectangle(height=2, width=3, 
		fill_color=color, fill_opacity=1, color=color)
	inner_rectangle = Rectangle(height=1.6, width=2.5, 
		fill_color=DARK_GRAY, fill_opacity=1, color=DARK_GRAY)
	extension = Rectangle(height=0.2, width=0.4, 
		fill_color=color, fill_opacity=1, color=color)
	extension.move_to(DOWN * (outer_rectangle.get_height() / 2 + extension.get_height() / 2))
	base = Rectangle(height=0.2, width=1,
		fill_color=color, fill_opacity=1, color=color)
	base.move_to(extension.get_center() + DOWN * extension.get_height())

	computer = VGroup(outer_rectangle, extension, base)

	left_circle = Circle(radius=0.27, color=color)
	left_circle.shift(LEFT * 0.6 + UP * 0.3)
	inner_left = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_left.shift(LEFT * 0.52, UP * 0.22)

	right_circle = Circle(radius=0.27, color=color)
	inner_right = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_right.shift(RIGHT * 0.52, UP*0.22)
	right_circle.shift(RIGHT * 0.6 + UP * 0.3)

	left_line = Line(DOWN * 0.3, DOWN * 0.5)
	right_line = Line(DOWN * 0.3, DOWN * 0.5)
	left_line.shift(LEFT * 0.5)
	right_line.shift(RIGHT * 0.5)
	bottom_line = Line(left_line.get_end(), right_line.get_end())
	left_line.set_color(color)
	right_line.set_color(color)
	bottom_line.set_color(color)

	smile = ArcBetweenPoints(left_line.get_start(), right_line.get_start())
	smile.set_color(color)
	
	left_eye_brow = ArcBetweenPoints(LEFT * 0.8 + UP * 0.6, LEFT * 0.4 + UP * 0.6, angle=-TAU/4)
	left_eye_brow.set_color(color)
	right_eye_brow = left_eye_brow.copy()
	right_eye_brow.shift(RIGHT * 1.2)
	right_eye_brow.set_color(color)

	eyes_and_smile = VGroup(left_circle, inner_left, right_circle, inner_right,
		smile, left_eye_brow, right_eye_brow)

	character = VGroup(computer, inner_rectangle, eyes_and_smile)
	character.scale(scale)
	character.move_to(position)


	return character

def create_confused_char(color=BLUE, scale=1, position=ORIGIN):
	outer_rectangle = Rectangle(height=2, width=3, 
		fill_color=color, fill_opacity=1, color=color)
	inner_rectangle = Rectangle(height=1.6, width=2.5, 
		fill_color=DARK_GRAY, fill_opacity=1, color=DARK_GRAY)
	extension = Rectangle(height=0.2, width=0.4, 
		fill_color=color, fill_opacity=1, color=color)
	extension.move_to(DOWN * (outer_rectangle.get_height() / 2 + extension.get_height() / 2))
	base = Rectangle(height=0.2, width=1,
		fill_color=color, fill_opacity=1, color=color)
	base.move_to(extension.get_center() + DOWN * extension.get_height())

	computer = VGroup(outer_rectangle, extension, base)

	left_circle = Circle(radius=0.27, color=color)
	left_circle.shift(LEFT * 0.6 + UP * 0.3)
	inner_left = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_left.shift(LEFT * 0.52, UP * 0.22)

	right_circle = Circle(radius=0.27, color=color)
	inner_right = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_right.shift(RIGHT * 0.52, UP*0.22)
	right_circle.shift(RIGHT * 0.6 + UP * 0.3)

	left_line = Line(DOWN * 0.3, DOWN * 0.5)
	right_line = Line(DOWN * 0.3, DOWN * 0.5)
	left_line.shift(LEFT * 0.5)
	right_line.shift(RIGHT * 0.5)
	bottom_line = Line(left_line.get_end(), right_line.get_end())
	left_line.set_color(color)
	right_line.set_color(color)
	bottom_line.set_color(color)

	smile = ArcBetweenPoints(left_line.get_start() + DOWN * 0.2, right_line.get_start(), angle=-TAU/4)
	smile.set_color(color)
	
	left_eye_brow = ArcBetweenPoints(LEFT * 0.8 + UP * 0.7, LEFT * 0.4 + UP * 0.7, angle=-TAU/4)
	left_eye_brow.set_color(color)
	right_eye_brow = ArcBetweenPoints(RIGHT * 0.8 + UP * 0.72, RIGHT * 0.4 + UP * 0.72, angle=-TAU/4)

	right_eye_brow.set_color(color)


	eyes_and_smile = VGroup(left_circle, inner_left, right_circle, inner_right,
		smile, left_eye_brow, right_eye_brow)

	character = VGroup(computer, inner_rectangle, eyes_and_smile)
	character.scale(scale)
	character.move_to(position)



	return character



class Container(Scene):

      def construct(self):
        start_time = time.time()
        particles = []
        
        num_particles = 100
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=3.0, width=3.0)
        box.shift(DOWN*0.15,LEFT*0.15)

        graph = Axes(
            
            x_length=13,
            y_length=7,
            axis_config={
                'color' : WHITE,
                'stroke_width' : 4,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : True,
                    'color' : WHITE
                }
            },

        )
        

        
        self.wait()
        wave = graph.get_graph(lambda x: np.sin(x),color = BLUE)
        wave.scale(0.3)
        wave.shift(LEFT*4.5)

        point1 = box
        point2 = wave
        d_arrow = DoubleArrow(start=point1, end=point2)
        d_arrow.scale(0.55)
        partic=TexMobject(r"E_{photon}=h \nu")
        partic.scale(0.8)
        partic.next_to(box, DOWN*3)

        wav=TextMobject("light beam")
        wav.scale(0.8)
        wav.next_to(wave, UP)

        velocities = [UP * i + RIGHT * j for i, j in zip(np.random.uniform(-0.2, 0.2, num_particles), np.random.uniform(-0.2, 0.2, num_particles))]
        positions = []
        for i in np.arange(-1.5, 1.5, 0.3):
            for j in np.arange(-1.5, 1.5, 0.3):
                positions.append(RIGHT * i + UP * j)
        for i in range(len(positions)):
            particle = Ball(radius=0.03)
            
            particle.id = i
            particle.move_to(positions[i])
            
            particles.append(particle)
        
        self.play(
            FadeIn(box),
            
        )
        self.play(
            *[FadeIn(particle) for particle in particles]
        )
        self.wait()
        self.play(
            ShowCreation(wave),
            Write(wav)
        )
        self.wait()
        default_char = create_computer_char(color=BLUE, scale=0.7, position= RIGHT * 4.5)
		    

        self.wait(3)
        self.play(FadeIn(default_char, RIGHT))
        self.play(
            FadeOut(wave),
            FadeOut(wav)
        )
        self.play(
            Write(partic)
        )


        start_time = time.time()
        particles = []
        num_particles = 6
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box2 = Box(height=3.5, width=3.5).set_color(YELLOW)
        shift_right = LEFT * 4
        box2.shift(shift_right)
        velocities = [
        LEFT * 0.5 + UP * 0.5, RIGHT * 0.5, LEFT + DOWN * 0.5, 
        RIGHT + DOWN * 0.5, RIGHT * 0.25 + DOWN * 0.25, RIGHT * 0.25 + UP * 0.25
        ]
        positions = [
        LEFT * 1 + UP * 0.5, UP * 0.5, RIGHT * 1 + UP * 0.5,
        LEFT * 1 + DOWN * 0.5, DOWN * 0.5, RIGHT * 1 + DOWN * 0.5,
        ]

        
        colors = [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE]
        radius =[0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box2),

        )
        self.play(
            *[FadeIn(particle) for particle in particles]
        )

        wav=TextMobject("Energy packets")
        wav.scale(0.8)
        wav.next_to(box2, UP)

        waves=TextMobject("photon")
        waves.scale(0.8)
        waves.next_to(box2, DOWN)

        self.play(
            Write(wav),
            Write(waves)
            )
        


        
class Recall(Scene):
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 6
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(YELLOW)
        shift_right = RIGHT * 4
        box.shift(shift_right)
        velocities = [
        LEFT * 1 + UP * 1, RIGHT * 1, LEFT + DOWN * 1, 
        RIGHT + DOWN * 1, RIGHT * 0.5 + DOWN * 0.5, RIGHT * 0.5 + UP * 0.5
        ]
        positions = [
        LEFT * 2 + UP * 1, UP * 1, RIGHT * 2 + UP * 1,
        LEFT * 2 + DOWN * 1, DOWN * 1, RIGHT * 2 + DOWN * 1,
        ]
        colors = [WHITE, MAROON, TEAL, GOLD, LIGHT_BROWN, PURPLE_A]
        radius =[0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box)
        )

        def update_particles(particles, dt):
            for i in range(len(particles)):
                particle = particles[i]
                particle.acceleration = np.array((0, 0, 0))
                particle.velocity = particle.velocity + particle.acceleration * dt
                particle.shift(particle.velocity * dt)
                handle_collision_with_box(particle, box, dt)
            
            handle_particle_collisions_opt(particles, dt)

        def handle_collision_with_box(particle, box, dt):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                particle.velocity[1] = -particle.velocity[1]
                particle.shift(particle.velocity * dt)
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() * BOX_THRESHOLD or \
                    particle.get_right_edge() >= box.get_right_edge() * BOX_THRESHOLD :
                particle.velocity[0] = -particle.velocity[0]
                particle.shift(particle.velocity * dt)

        def handle_particle_collisions(particles):
            for particle in particles:
                for other_particle in particles:
                    if particle.id != other_particle.id:
                        dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                        if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                            # ball.set_color(random.choice(t_colors))
                            # other_ball.set_color(random.choice(t_colors))
                            v1, v2 = get_response_velocities(particle, other_particle)
                            particle.velocity = v1
                            other_particle.velocity = v2

        def handle_particle_collisions_opt(particles, dt):
            possible_collisions = find_possible_collisions(particles) 
            # print([(a.id, b.id) for a, b in possible_collisions])
            # print(len(possible_collisions))
            for particle, other_particle in possible_collisions:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
                        particle.shift(particle.velocity * dt)
                        other_particle.shift(other_particle.velocity * dt)
        
        def find_possible_collisions(particles):
            # implements the sort and sweep algorithm for broad phase
            # helpful reference: https://github.com/mattleibow/jitterphysics/wiki/Sweep-and-Prune
            axis_list = sorted(particles, key=lambda x: x.get_left()[0])
            active_list = []
            possible_collisions = set()
            for particle in axis_list:
                to_remove = [p for p in active_list if particle.get_left()[0] > p.get_right()[0]]
                for r in to_remove:
                    active_list.remove(r)
                for other_particle in active_list:
                    possible_collisions.add((particle, other_particle))

                active_list.append(particle)
            
            return possible_collisions

        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        def show_all_collisions(particles):
            transforms = []
            end_positions = []
            all_pairs = []
            rows = [UP * 2, UP * 1, UP * 0, DOWN * 1, DOWN * 2]
            columns = [LEFT * 5.5, LEFT * 4, LEFT * 2.5]
            for col in columns:
                for row in rows:
                    end_positions.append(row + col)
            seen = set()
            i = 0
            for p1 in particles:
                for p2 in particles:
                    if p1.id == p2.id or (p2.id, p1.id) in seen:
                        continue
                    pair = VGroup(p1, p2)
                    p1_c = p1.copy().scale(0.7)
                    p2_c = p2.copy().scale(0.7)

                    p2_c.next_to(p1_c, RIGHT, buff=SMALL_BUFF)
                    transform_pair = VGroup(p1_c, p2_c).move_to(end_positions[i])
                    all_pairs.append(transform_pair)

                    transforms.append(TransformFromCopy(pair, transform_pair))
                    i += 1
                    seen.add((p1.id, p2.id))

            return transforms, all_pairs

        particles = VGroup(*particles)
        self.play(
            FadeIn(particles)
        )
        particles.add_updater(update_particles)
        self.add(particles)
        self.wait(16)
        particles.clear_updaters()

        print("--- %s seconds ---" % (time.time() - start_time))
        
        
        

        idea = TextMobject("Recall - Properties of particles").scale(1)
        idea.move_to(UP * 3).to_edge(LEFT * 2)
        h_line = Line(LEFT * 7.5, LEFT * 0.50)
        indent = LEFT * 1.5
        h_line.next_to(idea, DOWN).to_edge(indent)

        
        self.play(
            Write(idea),
            ShowCreation(h_line)
        )

        self.wait()

        topics = BulletedList(
            "Mass",
            "Well-defined Position",
            "Linear Momentum",
            "Countable",
            ).scale(0.9)

        topics.next_to(idea, DOWN).shift(DOWN * 1.0)
        for i in range(len(topics)):
            self.play(
                Write(topics[i])
            )
            self.wait(3)

        self.play(
            *[FadeOut(p) for p in particles],
        )
    def wavelength(self):
        graph = Axes(
            
            x_length=13,
            y_length=7,
            axis_config={
                'color' : WHITE,
                'stroke_width' : 4,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : True,
                    'color' : WHITE
                }
            },

        )
        title = TextMobject("Recall")
        title.set_color(YELLOW)
        title.scale(1.2)
        title.shift(UP * 3.5)
        self.h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        self.h_line.next_to(title, DOWN)

        self.play(
                Write(title),
                ShowCreation(self.h_line)
        )
        self.wait()
        wave = graph.get_graph(lambda x: np.sin(x),color = BLUE)
        wave.scale(0.5)
        point1 = wave.get_point_from_function(0)
        point2 = wave.get_point_from_function((2*np.pi))
        d_arrow = DoubleArrow(start=point1, end=point2)
        d_arrow.scale(0.55)
        d_arrow.shift((1.25*np.pi)*LEFT, 1*UP)
        
        lamb = TexMobject("\\lambda")
        lamb.next_to(d_arrow, UP)
        
        self.play(ShowCreation(wave), ShowCreation(d_arrow), Write(lamb))
        self.wait(4)
        self.play(FadeOut(wave), FadeOut(d_arrow), FadeOut(lamb))
    



      
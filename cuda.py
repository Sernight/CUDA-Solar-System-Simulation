# Mateusz Szmyd 179920
import random as rand
from vpython import *
import datetime
import time
from ctypes import *
import csv
import atexit

date = datetime.datetime(2021, 12, 29)  # początek symulacji
path_file = './data.csv'                # ścieżka do danych obiektów
AU = 149597870.700                      # km
duration = None                         # czas symulacji
dt = 60                                 # krok symulacji w sekundach
FPS = 600                               # maksymalne dopuszczalne FPS
trail = 0.01                            # rozmiar rysowanej orbity
original = False                        # czy skala ma być zachowana
names = []                              # lista wszystkich obiektów (ich nazw)
full_list = False                       # wybór pełnej listy do obserwacji


# klasy c_type
class Point3f(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("z", c_float)]


class Vector3f(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("z", c_float)]


class Values(Structure):
    _fields_ = [("position", Point3f),
                ("velocity", Vector3f)]

    def __init__(self, p, v):
        Structure.__init__(self, p, v)


class Orb(Structure):
    _fields_ = [("mass", c_float),
                ("radius", c_float),
                ("density", c_float),
                ("values", Values),
                ("distance", c_float)]

    def __init__(self, m, r, d, p, v, dd=0.):
        val = Values(p, v)
        super(Orb, self).__init__(m, r, d, val, dd)


# klasa pythonowa - tworzenie kul, rysowanie ich
class PyOrb:
    def __init__(self, n, m, r, d, p, v, dd, orb, c=(0, 0, 0)):
        self.name = n
        self.c_orb = pointer(orb)
        if c == (0, 0, 0):
            self.color = vector(rand.uniform(0, 1), rand.uniform(0, 1), rand.uniform(0, 1))
        else:
            self.color = vector(c[0], c[1], c[2])
        if not original:
            if self.name != 'Sun':
                self.sphere = sphere(pos=vector(self.c_orb.contents.values.position.x / AU,
                                                self.c_orb.contents.values.position.z / AU,
                                                -self.c_orb.contents.values.position.y / AU),
                                     radius=self.c_orb.contents.radius / AU * 100,
                                     color=self.color,
                                     make_trail=True,
                                     trail_radius=trail)
            else:
                self.sphere = sphere(pos=vector(self.c_orb.contents.values.position.x / AU,
                                                self.c_orb.contents.values.position.z / AU,
                                                -self.c_orb.contents.values.position.y / AU),
                                     radius=self.c_orb.contents.radius / AU * 60,
                                     color=self.color,
                                     make_trail=True,
                                     trail_radius=trail)
        else:
            self.sphere = sphere(pos=vector(self.c_orb.contents.values.position.x / AU * 100,
                                            self.c_orb.contents.values.position.z / AU * 100,
                                            -self.c_orb.contents.values.position.y / AU * 100),
                                 radius=self.c_orb.contents.radius / AU * 100,
                                 color=self.color,
                                 make_trail=True,
                                 trail_radius=trail)


# klasa całego układu z listą obiektów
class SolarSystem:
    def __init__(self):
        self.objects = []
        self.c_orbs = []
        self.time = 0.
        self.timestamp = 'Date: '

    # wczytanie obiektów z pliku
    def add_orbs(self, path=path_file):
        global names
        colors = [(1., 0.9, 0), (1., 0.6, 0.2), (1., 0.9, 0.6), (0.4, 0.7, 0.6), (1., 0.6, 0.2),
                  (0.9, 0.8, 0.7), (0.8, 0.8, 0.6), (0.5, 0.9, 1), (0.2, 0., 0.9), (0.9, 0.7, 0.6)]
        try:
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for count, row in enumerate(csv_reader):
                    names.append(row[0])
                    if count > 10:
                        c_orb = Orb(float(row[1]), float(row[2]), float(row[3]),
                                    (float(row[4]), float(row[5]), float(row[6])),
                                    (float(row[7]), float(row[8]), float(row[9])), float(row[10]))
                        orb = PyOrb(row[0], float(row[1]), float(row[2]), float(row[3]),
                                    (float(row[4]), float(row[5]), float(row[6])),
                                    (float(row[7]), float(row[8]), float(row[9])), float(row[10]), c_orb)
                        self.objects.append(orb)
                        self.c_orbs.append(c_orb)
                    else:
                        if count != 0:
                            c_orb = Orb(float(row[1]), float(row[2]), float(row[3]),
                                        (float(row[4]), float(row[5]), float(row[6])),
                                        (float(row[7]), float(row[8]), float(row[9])), float(row[10]))
                            orb = PyOrb(row[0], float(row[1]), float(row[2]), float(row[3]),
                                        (float(row[4]), float(row[5]), float(row[6])),
                                        (float(row[7]), float(row[8]), float(row[9])), float(row[10]), c_orb,
                                        colors[count - 1])
                            self.objects.append(orb)
                            self.c_orbs.append(c_orb)
        except csv.Error as e:
            print(e)
        self.c_orbs = (Orb * len(self.objects))(*self.c_orbs)
        scene.title = self.timestamp

    # aktualizacja obrazu
    def update_scene(self):
        self.time += dt
        if not original:
            for i, obj in enumerate(self.objects):
                if self.c_orbs[i].mass != 0:
                    obj.sphere.pos = vector(self.c_orbs[i].values.position.x / AU,
                                            self.c_orbs[i].values.position.z / AU,
                                            -self.c_orbs[i].values.position.y / AU)
                else:
                    obj.sphere.clear_trail()
        else:
            for i, obj in enumerate(self.objects):
                if self.c_orbs[i].mass != 0:
                    obj.sphere.pos = vector(self.c_orbs[i].values.position.x / AU * 100,
                                            self.c_orbs[i].values.position.z / AU * 100,
                                            -self.c_orbs[i].values.position.y / AU * 100)
                else:
                    obj.sphere.clear_trail()
        scene.title = str(date)

    # obliczenia dla CPU
    def update_values(self):
        for i in range(len(self.objects)):
            ax = 0
            ay = 0
            az = 0
            for j in range(len(self.objects)):
                if i != j:
                    direction_x = self.c_orbs[j].values.position.x - self.c_orbs[i].values.position.x
                    direction_y = self.c_orbs[j].values.position.y - self.c_orbs[i].values.position.y
                    direction_z = self.c_orbs[j].values.position.z - self.c_orbs[i].values.position.z

                    ax += 6.6743015151515e-20 * direction_x * self.c_orbs[j].mass / \
                          distance(self.c_orbs[i].values.position, self.c_orbs[j].values.position) ** 3
                    ay += 6.6743015151515e-20 * direction_y * self.c_orbs[j].mass / \
                          distance(self.c_orbs[i].values.position, self.c_orbs[j].values.position) ** 3
                    az += 6.6743015151515e-20 * direction_z * self.c_orbs[j].mass / \
                          distance(self.c_orbs[i].values.position, self.c_orbs[j].values.position) ** 3

            self.c_orbs[i].values.position.x += self.c_orbs[i].values.velocity.x * dt
            self.c_orbs[i].values.position.y += self.c_orbs[i].values.velocity.y * dt
            self.c_orbs[i].values.position.z += self.c_orbs[i].values.velocity.z * dt

            self.c_orbs[i].values.velocity.x += ax * dt
            self.c_orbs[i].values.velocity.y += ay * dt
            self.c_orbs[i].values.velocity.z += az * dt

    # dodawanie nowego obiektu w czasie działania programu
    def add_orb(self, name: str, m, r, d, p, v):
        c_orb = Orb(m, r, d, p, v)
        orb = PyOrb(name, m, r, d, p, v, 0, c_orb)
        self.objects.append(orb)
        self.c_orbs = (Orb * len(self.objects))(*self.c_orbs)
        self.c_orbs[len(self.c_orbs) - 1] = c_orb


def distance(a: Point3f, b: Point3f):
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def terminal_handling():
    global original, full_list
    answer = input("Show full list of all bodies to focus [y/n]: ")
    while answer != 'y' and answer != 'n':
        print('Try again. Enter "y" or "n"')
        answer = input("Show full list of all bodies to focus [y/n]: ")
    if answer == 'y':
        full_list = True
    else:
        if answer == 'n':
            full_list = False
    answer = input("Original or scaled system [o/s]: ")
    while answer != 'o' and answer != 's':
        print('Try again. Enter "o" or "s"')
        answer = input("Original or scaled system [o/s]: ")
    if answer == 'o':
        original = True
        trail = 0.0006
    else:
        if answer == 's':
            original = False
            answer = input("Visible trails of orbits from distance [y/n]: ")
            while answer != 'y' and answer != 'n':
                print('Try again. Enter "y" or "n"')
                answer = input("Visible trails of orbits from distance [y/n]: ")
            if answer == 'y':
                trail = 0.007
            else:
                if answer == 'n':
                    trail = 0.001
    answer = input("Time of simulation in days: ")
    while not answer.isdecimal():
        print('Try again. Enter number')
        answer = input("Time of simulation in days: ")
    duration = date + datetime.timedelta(days=int(answer))
    return trail, duration


running = True


# zatrzymanie/uruchomienie animacji
def run(b):
    global running, dt, remember_dt
    running = not running
    if running:
        b.text = 'Pause'
        dt = remember_dt
    else:
        b.text = 'Run'
        remember_dt = dt
        dt = 0
    return


# zmiana skali systemu
def scaling(b):
    global original, trail
    original = not original
    if original:
        b.text = 'Scaled'
        ss.objects[0].sphere.radius = ss.c_orbs[0].radius / AU * 100
        for obj in ss.objects:
            obj.sphere.clear_trail()
        trail = 0.0006
    else:
        b.text = 'Original'
        ss.objects[0].sphere.radius = ss.c_orbs[0].radius / AU * 60
        for obj in ss.objects:
            obj.sphere.clear_trail()
        trail = 0.004
    return


def add_orb(b):
    mass = input("Mass of object [kg]: ")
    # while not mass.isnumeric():
    #     print('Try again. Enter number')
    #     mass = input("Mass of object [kg]: ")

    radius = input("Radius of object [km]: ")
    # while not radius.isnumeric():
    #     print('Try again. Enter number')
    #     radius = input("Radius of object [km]: ")

    density = input("Density of object [g/cm^3]: ")
    # while not density.isnumeric():
    #     print('Try again. Enter number')
    #     density = input("Density of object [g/cm^3]: ")

    posx = input("Position x of object [km]: ")
    # while not posx.isnumeric():
    #     print('Try again. Enter number')
    #     posx = input("Position x of object [km]: ")
    posy = input("Position y of object [km]: ")
    # while not posy.isnumeric():
    #     print('Try again. Enter number')
    #     posy = input("Position y of object [km]: ")
    posz = input("Position z of object [km]: ")
    # while not posz.isnumeric():
    #     print('Try again. Enter number')
    #     posz = input("Position z of object [km]: ")

    velx = rand.uniform(-1, 1)
    vely = rand.uniform(-1, 1)
    velz = rand.uniform(-1, 1)
    try:
        ss.add_orb(("O" + str(len(ss.objects) + 1)), float(mass), float(radius), float(density),
                   (float(posx), float(posy), float(posz)), (velx, vely, velz))

        m.choices += str("O" + str(len(ss.objects) + 1))
    except Exception as e:
        print(e)
        print("Try again.")


# wybór środka sceny
def focus(m):
    selected = m.selected
    scale = AU
    if not original:
        scale = AU
    else:
        scale = AU / 100

    for i, obj in enumerate(ss.objects):
        if obj.name == selected:
            scene.center = vector(ss.c_orbs[i].values.position.x / scale, ss.c_orbs[i].values.position.z / scale,
                                  -ss.c_orbs[i].values.position.y / scale)


def configure_scene():
    global original
    scene.height = 800
    scene.width = 1800
    scene.center = vector(0, 0, 0)
    scene.camera.pos = vector(-63.0147, 32.4819, 10.8686)
    scene.camera.axis = vector(63.0147, -32.4819, -10.8686)
    scene.append_to_caption('\n')
    pause_button = button(text="Pause", pos=scene.caption_anchor, bind=run)
    button(text=("Scaled" if original else "Original"), pos=scene.caption_anchor, bind=scaling)
    # add_button = button(text="Add body", pos=scene.caption_anchor, bind=add_orb)
    scene.append_to_caption('\n \n Center of scene: ')
    if full_list:
        m = menu(choices=names, bind=focus)
    else:
        m = menu(choices=['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
                          'Uranus', 'Neptune', 'Pluto'],
                 bind=focus)
    return pause_button, m


# funkcja wywoływana przez atexit
def close():
    gpu.freeMem()
    with open('./times.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(times)

    # with open('./positions.csv', 'w') as f:
    #     write = csv.writer(f)
    #     for i, obj in enumerate(ss.c_orbs):
    #         write.writerow([i, obj.values.position.x, obj.values.position.y, obj.values.position.z])


times = []

atexit.register(close)

if __name__ == '__main__':
    gpu = cdll.LoadLibrary('./libKernel.so')
    ss = SolarSystem()

    trail, duration = terminal_handling()
    ss.add_orbs()
    pause_button, m = configure_scene()
    while date < duration:
        rate(FPS)
        if dt != 0:
            start = time.time()
            ss.time += dt
            date += datetime.timedelta(seconds=dt)
            gpu.update(ss.c_orbs, len(ss.c_orbs), c_float(dt))      # GPU
            # ss.update_values()                                      # CPU
            ss.update_scene()
            focus(m)
            # zapisanie czasu jednej klatki (raz na dobę symulacji)
            # if ss.time % (24 * 3600) == 0:
            #     times.append([date, time.time() - start])
        else:
            focus(m)
    else:
        gpu.freeMem()
        rate(FPS)

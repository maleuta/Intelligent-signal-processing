# Laboratorium: MeanShift, CamShit, OpticalFlow, Filtr Kalamana

## Inteligentne przetwarzanie sygnałów

- Maksymalna liczba punktów: 10
- Skala ocen za punkty:
    - 9-10 ~ bardzo dobry (5.0)
    - 8 ~ plus dobry (4.5)
    - 7 ~ dobry (4.0)
    - 6 ~ plus dostateczny (3.5)
    - 5 ~ dostateczny (3.0)
    - 0-4 ~ niedostateczny (2.0)

Skorzystamy z pliku car_bike_track.mp4.
Niektóre z algorytmów, z których będziemy korzystali, wymagają wykrycia obiektu (dokonania pomiaru jego położenia).
Na naszym nagraniu widoczne są trzy obiekty: czerwony, zielony i niebieski. Dla uproszczenia wykorzystamy tę informację w celu ich detekcji. Możemy jej dokonać na przykład przy pomocy progowania, poniższym programem.

```python

def detect_bounding_box(my_frame, my_tolerance):
    my_mask = np.zeros_like(my_frame).astype(np.uint8)
    map_blue = (my_tolerance[0][0] < my_frame[:,:,0]) & (my_frame[:,:,0] < my_tolerance[0][1])
    map_green = (my_tolerance[1][0] < my_frame[:, :, 1]) & (my_frame[:, :, 1] < my_tolerance[1][1])
    map_red = (my_tolerance[2][0] < my_frame[:, :, 2]) & (my_frame[:, :, 2] < my_tolerance[2][1])
    my_mask[map_red & map_green & map_blue] = 255
    y, x = np.where(my_mask[:,:,0].astype(bool))
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    return my_mask, xmin, ymin, xmax - xmin, ymax - ymin

```

Przykładowe wywołanie powyższej metody dla trzech kolorów obiektów:

```python

# zielony
my_mask, x, y, w, h = detect_bounding_box(frame, [[0, 64], [180, 256], [180, 256]])
track_window_green = (x, y, w, h)

# niebieski
my_mask, x, y, w, h = detect_bounding_box(frame, [[128,256],[0,64],[0,64]])
track_window_blue = (x, y, w, h)

# czerwony
my_mask, x, y, w, h = detect_bounding_box(frame, [[0, 64], [0, 64], [212, 256]])
track_window_red = (x, y, w, h)

```

Kiedy mamy już okna, w których znajdują się obiekty, możemy przejść do wykonania zadań :-)

[2 punkty] Napisz program, który przy pomocy algorytmu Mean Shift będzie śledził położenie każdego z trzech obiektów na nagraniu car_bike_track.mp4. Zaznaczaj położenie każdego z obiektów przy pomocy ramki (bounding box). Warto to zrobić w następujący sposób: stwórz klasę, której poszczególne instancje będą odpowiadały za śledzenie każdego z obiektów z osobna. Na tej samej zasadzie warto zrealizować pozostałe zadania.

[2 punkty] Napisz program, który przy pomocy algorytmu Cam Shift będzie śledził położenie każdego z trzech obiektów na nagraniu car_bike_track.mp4. Zaznaczaj położenie każdego z obiektów przy pomocy ramki (bounding box).
 UWAGA! Ponieważ wielkość okna w Cam Shift może się zmienić, algorytm może stać się niestabilny i ramka może się niekontrolowanie rozrosnąć. Aby temu zapobiec, można zbudować histogram w oparciu o wartości HS:

```python

class camshift:
    def __init__(self, frame, track_window):
        self.track_window = track_window
        (x, y, w, h) = track_window
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        #roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180]) - this uses onlu H
        self.roi_hist = cv.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv.normalize(self.roi_hist, self.roi_hist, 0, 255, cv.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 1)

    def calculate(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1) - this uses onlu H
        dst = cv.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
        # apply camshift to get the new location
        ret, self.track_window = cv.CamShift(dst, self.track_window, self.term_crit)
        return ret

```

[2 punkty] Napisz program, który przy pomocy algorytmu Lucas-Kanade Optical Flow zwizualizuje ruch obiektów na nagraniu.

[2 punkty] Napisz program, który przy pomocy algorytmu Farneback Optical Flow zwizualizuje ruch obiektów na nagraniu. Pokaż, jak wyglądają estymacje wektorów ruchu.

[2 punkty] Napisz program, który dokona predykcji pozycji obiektów przy pomocy dwuwymiarowego filtru Kalmana.

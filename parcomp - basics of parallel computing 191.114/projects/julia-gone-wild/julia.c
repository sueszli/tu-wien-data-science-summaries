#include <stdio.h>

enum { WIDTH = 240, HEIGHT = 160, STEPS = 100 };
static int pixels[HEIGHT][WIDTH];

static int escape_steps(int x, int y)
{
    double re = (x - WIDTH / 2) / 80.0;
    double im = (y - HEIGHT / 2) / 80.0;
    int step = 0;
    for (; step < STEPS && re * re + im * im <= 4.0; ++step) {
        double next_re = re * re - im * im - 0.1;
        im = 2 * re * im + 0.65;
        re = next_re;
    }
    return step;
}

int main(void)
{
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            pixels[y][x] = escape_steps(x, y);

    printf("P2\n%d %d\n%d\n", WIDTH, HEIGHT, STEPS);
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            printf("%d%c", pixels[y][x], x == WIDTH - 1 ? '\n' : ' ');
    return 0;
}

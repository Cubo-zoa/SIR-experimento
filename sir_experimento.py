"""
Experimento SIR con dulces para un grupo de N personas (N par, por defecto 46).

Hace tres cosas:
  1. Calendario de emparejamientos: en cada ronda todos tienen exactamente una
     pareja, sorteada al azar de antemano (con semilla, para poder proyectarla).
     Guarda CSVs: por pareja y por persona.
  2. Simulación del experimento (Monte Carlo) con las mismas reglas que se
     usarán en vivo, comparada con la curva "promedio" (determinista).
  3. Estimadores de beta y gamma a partir de la tabla S, I, R que anota el
     voluntario después de cada ronda.

Reglas del juego (por ronda):
  - Contagio: si la pareja es S-I, el susceptible tira un d6; se contagia con 1..4  (p = 2/3).
  - Recuperación: cada infectado (al inicio de la ronda) tira un d6; se
    recupera con 1..2 (q = 1/3).
  - Primero se resuelven los contagios con el estado del INICIO de la ronda;
    los recién contagiados no tiran para recuperarse en esa misma ronda.

Verdad por diseño: beta = p * N/(N-1) ≈ p,  gamma = q,  R0 = beta/gamma ≈ 2.

Uso:
    pip install numpy matplotlib
    python sir_experimento.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ parámetros
N = 46            # personas (par)
I0 = 5            # infectados iniciales (varias "semillas" evitan que muera al inicio)
P_CONTAGIO = 2/3  # d6: 1-4
Q_RECUP = 1/3     # d6: 1-2
RONDAS = 15       # rondas máximas a jugar (el brote suele cerrar antes)
N_SIM = 1000      # corridas de Monte Carlo
SEED = 7


# ----------------------------------------------------- 1. calendario de parejas
def calendario_aleatorio(n, n_rondas, rng):
    """Devuelve lista de rondas; cada ronda es una lista de n/2 parejas (a, b)
    con personas numeradas 0..n-1, formadas al azar.

    Nota: se probó el método del círculo (round-robin, sin repetir pareja) y
    sesga el contagio a la baja (la estructura fija correlaciona quién se
    encuentra con quién); con emparejamientos aleatorios, beta sale sin sesgo.
    Alguna pareja puede repetirse entre rondas; no afecta al modelo."""
    assert n % 2 == 0, "N debe ser par"
    rondas = []
    for _ in range(n_rondas):
        p = rng.permutation(n)
        rondas.append([(int(p[2 * i]), int(p[2 * i + 1])) for i in range(n // 2)])
    return rondas


def guardar_calendario(rondas, n_rondas, prefijo="calendario"):
    """CSV 1: una fila por pareja. CSV 2: una fila por persona, una columna por
    ronda con el número de su pareja (útil para proyectar o imprimir)."""
    n = len(rondas[0]) * 2
    with open(f"{prefijo}_parejas.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ronda", "persona_A", "persona_B"])
        for r in range(n_rondas):
            for a, b in rondas[r]:
                w.writerow([r + 1, a + 1, b + 1])  # numeración 1..N para humanos

    pareja = np.zeros((n, n_rondas), dtype=int)
    for r in range(n_rondas):
        for a, b in rondas[r]:
            pareja[a, r], pareja[b, r] = b + 1, a + 1
    with open(f"{prefijo}_por_persona.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["persona"] + [f"R{r+1}" for r in range(n_rondas)])
        for i in range(n):
            w.writerow([i + 1] + list(pareja[i]))


# --------------------------------------------------------------- 2. simulación
S, I, R = 0, 1, 2


def simular(rondas, rng, n=N, i0=I0, p=P_CONTAGIO, q=Q_RECUP, n_rondas=RONDAS):
    """Una corrida del experimento. Devuelve arreglo (T+1, 3) con S, I, R por ronda.
    Se reasignan las etiquetas al azar para que el calendario no sesgue nada."""
    perm = rng.permutation(n)
    estado = np.full(n, S)
    estado[rng.choice(n, i0, replace=False)] = I
    hist = [[(estado == S).sum(), (estado == I).sum(), (estado == R).sum()]]

    for r in range(n_rondas):
        nuevo = estado.copy()
        # contagios (usando el estado al inicio de la ronda)
        for a, b in rondas[r]:
            a, b = perm[a], perm[b]
            for x, y in ((a, b), (b, a)):
                if estado[x] == S and estado[y] == I and rng.random() < p:
                    nuevo[x] = I
        # recuperaciones (solo los que eran I al inicio)
        for i in np.where(estado == I)[0]:
            if rng.random() < q:
                nuevo[i] = R
        estado = nuevo
        hist.append([(estado == S).sum(), (estado == I).sum(), (estado == R).sum()])
        if hist[-1][1] == 0:
            break
    return np.array(hist)


def promedio_determinista(n=N, i0=I0, p=P_CONTAGIO, q=Q_RECUP, n_rondas=RONDAS):
    """Recursión de campo medio (la 'EDO' en tiempo discreto).
    Un susceptible se contagia con prob p * I/(n-1) por ronda."""
    s, i, r = n - i0, i0, 0.0
    out = [[s, i, r]]
    for _ in range(n_rondas):
        nuevos = s * p * i / (n - 1)
        rec = q * i
        s, i, r = s - nuevos, i + nuevos - rec, r + rec
        out.append([s, i, r])
    return np.array(out)


# ------------------------------------------------------------- 3. estimadores
def estimar(S_t, I_t, R_t, n=None):
    """Estimadores a partir de la tabla (S, I, R) de cada ronda, empezando en la ronda 0.
    Cuentan eventos en lugar de ajustar una curva:
        gamma = recuperaciones / suma de I_t
        beta  = contagios / suma de S_t * I_t / (n-1)
    (n-1 porque cada persona tiene una sola pareja entre las otras n-1.)"""
    S_t, I_t, R_t = map(np.asarray, (S_t, I_t, R_t))
    n = n or int(S_t[0] + I_t[0] + R_t[0])
    contagios = S_t[:-1] - S_t[1:]
    recups = R_t[1:] - R_t[:-1]
    gamma = recups.sum() / I_t[:-1].sum()
    beta = contagios.sum() / (S_t[:-1] * I_t[:-1] / (n - 1)).sum()
    # error estándar aproximado (conteos Poisson)
    se_gamma = gamma / np.sqrt(max(recups.sum(), 1))
    se_beta = beta / np.sqrt(max(contagios.sum(), 1))
    return beta, gamma, se_beta, se_gamma


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    rondas = calendario_aleatorio(N, RONDAS, np.random.default_rng(SEED))
    guardar_calendario(rondas, RONDAS)
    print(f"Calendario guardado: {RONDAS} rondas de {N // 2} parejas "
          f"(calendario_parejas.csv, calendario_por_persona.csv)")

    # verificación: todos tienen exactamente una pareja en cada ronda
    for ronda in rondas:
        assert sorted(x for par in ronda for x in par) == list(range(N))

    beta_real = P_CONTAGIO * N / (N - 1)
    gamma_real = Q_RECUP
    print(f"Verdad por diseño: beta = {beta_real:.3f}, gamma = {gamma_real:.3f}, "
          f"R0 = {beta_real / gamma_real:.2f}")

    # Monte Carlo
    corridas = [simular(rondas, rng) for _ in range(N_SIM)]
    morian = np.mean([N - c[-1, 0] < 0.4 * N for c in corridas])
    est = np.array([estimar(*c.T)[:2] for c in corridas if len(c) > 2])
    print(f"Corridas donde el brote casi no despega (<40% llegan a contagiarse): {morian:.1%}")
    print(f"beta  estimado: media {est[:,0].mean():.3f}, desv. est. {est[:,0].std():.3f}")
    print(f"gamma estimado: media {est[:,1].mean():.3f}, desv. est. {est[:,1].std():.3f}")

    # Una corrida "como la real" para el ejemplo de los estimadores
    ejemplo = corridas[0]
    b, g, sb, sg = estimar(*ejemplo.T)
    print(f"\nCorrida de ejemplo: beta = {b:.2f} ± {sb:.2f}, gamma = {g:.2f} ± {sg:.2f}, "
          f"R0 ≈ {b / g:.2f}")

    # Gráfica
    det = promedio_determinista()
    t_max = max(len(c) for c in corridas)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for c in corridas[:150]:
        ax[0].plot(c[:, 1], color="tab:red", alpha=0.07)
    ax[0].plot(ejemplo[:, 1], color="black", lw=2, label="una corrida (como la real)")
    ax[0].plot(det[:, 1], color="tab:blue", lw=2.5, ls="--", label="curva promedio (EDO)")
    ax[0].set(title="Infectados por ronda", xlabel="ronda", ylabel="I", xlim=(0, RONDAS))
    ax[0].legend()

    for k, (nombre, col) in enumerate([("S", "tab:green"), ("I", "tab:red"), ("R", "tab:gray")]):
        ax[1].plot(ejemplo[:, k], color=col, lw=2, label=nombre)
        ax[1].plot(det[:, k], color=col, lw=1.5, ls="--")
    ax[1].set(title="Corrida de ejemplo (línea) vs. promedio (punteada)",
              xlabel="ronda", ylabel="personas", xlim=(0, RONDAS))
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("sir_experimento.png", dpi=150)
    print("Gráfica guardada: sir_experimento.png")

    # ---- Para usar con los datos REALES del experimento, descomenta y llena:
    # S_real = [41, 38, 33, ...]   # S al inicio y después de cada ronda
    # I_real = [5, 7, 10, ...]
    # R_real = [0, 1, 3, ...]
    # print(estimar(S_real, I_real, R_real))

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

FC = 0.196  # frequência de corte (entre 0 e 0.5)
M_values = [40]  # diferentes comprimentos do filtro

plt.figure(figsize=(10, 5))

pb = []

for M in M_values:
    N = M + 1
    pb = np.zeros(N)

    for i in range(N):
        x = i - M / 2
        if x == 0:
            pb[i] = 2 * np.pi * FC
        else:
            pb[i] = np.sin(2 * np.pi * FC * x) / x
        # Janela de Hamming
        pb[i] *= 0.54 - 0.46 * np.cos(2 * np.pi * i / M)

    # Normaliza para ganho unitário em DC
    pb = pb / np.sum(pb)


FC = 0.204  # frequência de corte (entre 0 e 0.5)
M_values = [40]  # diferentes comprimentos do filtro

pa = []


for M in M_values:
    N = M + 1
    pa = np.zeros(N)

    for i in range(N):
        x = i - M / 2
        if x == 0:
            pa[i] = 2 * np.pi * FC
        else:
            pa[i] = np.sin(2 * np.pi * FC * x) / x
        # Janela de Hamming
        pa[i] *= 0.54 - 0.46 * np.cos(2 * np.pi * i / M)

    # Normaliza para ganho unitário em DC
    pa = pa / np.sum(pa)

    # Transforma para filtro passa-alta: h_pa[n] = δ[n] - h_pb[n]
    pa = -pa  # Inverte todos os valores
    pa[M // 2] += 1  # Adiciona 1 na posição central (impulso unitário)

rf = [a + b for a, b in zip(pb, pa)]

# Calcula resposta em frequência
w, H = signal.freqz(rf, worN=1024)
freq_norm = w / (2 * np.pi)

plt.plot(freq_norm, np.abs(H) / np.max(np.abs(H)), label=f"M = {M}")

plt.axvline(FC, color="r", linestyle="--", label=f"Frequência de corte (FC={FC})")
plt.title("Resposta em Frequência - Filtros Rejeita-Faixa (Vários Comprimentos)")
plt.xlabel("Frequência Normalizada (ciclos/amostra)")
plt.ylabel("Amplitude Normalizada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Salva os coeficientes do último filtro (M=100) em arquivo
np.save("output/coeficientes_h_pa.npy", rf)
print(
    f"Coeficientes h[n] do filtro passa-alta salvos em 'output/coeficientes_h_pa.npy'"
)
print(f"Comprimento do filtro: {len(rf)}")
print(f"Primeiros 5 coeficientes: {rf[:5]}")
print(f"Últimos 5 coeficientes: {rf[-5:]}")

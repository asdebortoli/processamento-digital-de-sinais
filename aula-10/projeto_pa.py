import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

FC = 0.14  # frequência de corte (entre 0 e 0.5)
M_values = [40]  # diferentes comprimentos do filtro

plt.figure(figsize=(10, 5))

for M in M_values:
    N = M + 1
    h = np.zeros(N)

    for i in range(N):
        x = i - M / 2
        if x == 0:
            h[i] = 2 * np.pi * FC
        else:
            h[i] = np.sin(2 * np.pi * FC * x) / x
        # Janela de Hamming
        h[i] *= 0.54 - 0.46 * np.cos(2 * np.pi * i / M)

    # Normaliza para ganho unitário em DC
    h = h / np.sum(h)

    # Transforma para filtro passa-alta: h_pa[n] = δ[n] - h_pb[n]
    h = -h  # Inverte todos os valores
    h[M // 2] += 1  # Adiciona 1 na posição central (impulso unitário)

    # Calcula resposta em frequência
    w, H = signal.freqz(h, worN=1024)
    freq_norm = w / (2 * np.pi)

    plt.plot(freq_norm, np.abs(H) / np.max(np.abs(H)), label=f"M = {M}")

plt.axvline(FC, color="r", linestyle="--", label=f"Frequência de corte (FC={FC})")
plt.title("Resposta em Frequência - Filtros Passa-Alta (Vários Comprimentos)")
plt.xlabel("Frequência Normalizada (ciclos/amostra)")
plt.ylabel("Amplitude Normalizada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Salva os coeficientes do último filtro (M=100) em arquivo
np.save("output/coeficientes_h_pa.npy", h)
print(
    f"Coeficientes h[n] do filtro passa-alta salvos em 'output/coeficientes_h_pa.npy'"
)
print(f"Comprimento do filtro: {len(h)}")
print(f"Primeiros 5 coeficientes: {h[:5]}")
print(f"Últimos 5 coeficientes: {h[-5:]}")

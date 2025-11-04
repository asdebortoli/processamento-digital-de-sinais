import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average_processing(sample_vector, coefficient_vector):
    """
    Simple moving average system: y[n] = a0* x[n] + a1* x[n-1] + a2* x[n-2] + a3* x[n-3]
    """
    output = 0
    for n in range(0, len(sample_vector)):
        output += coefficient_vector[n] * sample_vector[n]
    return output


def salvar_arquivo_pcm(dados, nome_arquivo):
    """
    Salva dados como arquivo PCM
    """
    try:
        # Converte de float para int16
        dados_int16 = (dados * 32767).astype(np.int16)

        # Salva como arquivo binário
        with open(nome_arquivo, "wb") as f:
            f.write(dados_int16.tobytes())

        print(f"💾 Sinal de saída salvo em '{nome_arquivo}'")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")
        return False


def main():
    # Carrega os coeficientes h[n] salvos pelo projeto.py
    try:
        coefficient_vector = np.load("output/coeficientes_h_pa.npy")
        print(f"✅ Coeficientes h[n] carregados com sucesso!")
        print(f"   • Comprimento do filtro: {len(coefficient_vector)}")
        print(f"   • Primeiros 5 coeficientes: {coefficient_vector[:5]}")
        print(f"   • Últimos 5 coeficientes: {coefficient_vector[-5:]}")
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'output/coeficientes_h.npy' não encontrado.")
        print("   Execute primeiro o projeto.py para gerar os coeficientes.")
        return
    except Exception as e:
        print(f"❌ Erro ao carregar coeficientes: {e}")
        return

    # Carrega o arquivo PCM de entrada
    try:
        # x = np.fromfile("input/sweep_20_3k4.pcm", dtype=np.int16)
        x = np.fromfile("input/seno_400.pcm", dtype=np.int16)
        print(f"📁 Arquivo PCM carregado: {len(x)} amostras")
        print(f"   • Duração aproximada: {len(x)/8000:.2f} segundos (fs=8kHz)")

        # Normaliza para float entre -1 e 1
        x = x.astype(np.float32) / 32768.0

    except FileNotFoundError:
        print("❌ Erro: Arquivo 'input/sweep_20_3k4.pcm' não encontrado!")

        # Lista arquivos disponíveis na pasta input
        if os.path.exists("input"):
            arquivos_input = os.listdir("input")
            if arquivos_input:
                print("📁 Arquivos disponíveis na pasta input:")
                for arquivo in arquivos_input:
                    print(f"   • {arquivo}")
            else:
                print("📁 Pasta input está vazia.")
        else:
            print("📁 Pasta input não existe.")
        return
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo: {e}")
        return

    K = len(coefficient_vector)
    sample_vector = np.zeros(K)
    y = np.zeros(len(x))

    for n in range(len(x)):
        sample_vector[1:] = sample_vector[:-1]
        sample_vector[0] = x[n]

        y[n] = moving_average_processing(sample_vector, coefficient_vector)

    # Salva o resultado como PCM
    nome_saida = "output/sinal_saida_filtrado.pcm"
    if salvar_arquivo_pcm(y, nome_saida):
        print(f"\n" + "─" * 50)
        print("✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("─" * 50)
        print(f"📁 Arquivo de entrada: input/sweep_20_3k4.pcm")
        print(f"📁 Arquivo de saída: {nome_saida}")
        print(f"🔧 Filtro aplicado: Passa-baixas com M={K-1}")

    # Print results with simple but pretty formatting
    print("\n" + "─" * 50)
    print("📊 MOVING AVERAGE RESULTS")
    print("─" * 50)
    print(f"🔧 K = {K}")
    print(f"📥 Input samples: {len(x)}")
    print(f"📤 Output samples: {len(y)}")
    print(f"📊 Input range: [{np.min(x):.4f}, {np.max(x):.4f}]")
    print(f"📊 Output range: [{np.min(y):.4f}, {np.max(y):.4f}]")
    print("─" * 50)

    # Plot the impulse response
    n = np.arange(len(x))

    plt.figure(figsize=(10, 6))

    # Plot input
    plt.subplot(2, 1, 1)
    plt.stem(n, x, "b-", label="Input x[n]")
    plt.title("Input Signal (Sweep)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot output
    plt.subplot(2, 1, 2)
    plt.stem(n, y, "r-", label="Output y[n]")
    plt.title("Output Signal (Filtered)")
    plt.xlabel("Sample n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

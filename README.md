# Treine Certo: Aplicação Para Análise Da Corretude De Exercícios Físicos Utilizando Aprendizado De Máquina

Departamento de Sistemas e Computação – FURB
Curso de Ciência da Computação
Trabalho de Conclusão de Curso– 2025/1

| **Acadêmico**                | **Coorientador**             | **Orientador**               |
|-----------------------------|------------------------------|------------------------------|
| Luiz Gustavo Klitzke        | Diego Rafael Eising          | Aurélio Faustino Hoppe       |
| lgklitzke@furb.br           | diegoskel@hotmail.com        | aureliof@furb.br             |


![Exemplo](./front/examples/deadlift_output.gif)

## 📌 Objetivo
A aplicação tem como objetivo oferecer uma **ferramenta de análise automática da execução de exercícios físicos** a partir de vídeos gravados com câmeras convencionais, auxiliando praticantes e profissionais da área na identificação de **erros técnicos e posturais** durante a realização dos movimentos.

## ⚙️ Funcionalidades

- 📸 Análise de vídeos com base em pontos-chave do corpo extraídos via [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- 🤖 Classificação de execuções em **corretas** ou com **erro técnico**
- 🧠 Modelo híbrido baseado em **GCN (Graph Convolutional Network)** + **LSTM** + **atenção temporal**
- 🖥️ Interface gráfica desktop intuitiva
- 📤 Exportação dos resultados em `.mp4` e `.json`


## Compilação e Instalação
Serão listados abaixo os passos para compilação e instalação da aplicação e suas dependências.

### Instalação do OpenPose
Primeiramente, é necessário realizar a compilação da API do OpenPose para Python.

O repositório oficial pode ser encontrado, juntamente com um tutorial para sua compilação em: https://github.com/CMU-Perceptual-Computing-Lab/openpose

Podem ocorrer vários erros no processo, uma vez que alguns componentes e links estão offline, da mesma forma, é necessário utilizar algumas dependências em versões específicas, como o próprio CMake e é necessário utilizar o Visual Studio 2017 para compilação final da solution gerada.

Segue downloads de alguns itens compatíveis, para auxiliar:
- python 3.7 = https://www.python.org/downloads/release/python-370/
- CMake 3.26 (cmake-3.26.0-rc1-windows-x86_64.msi) = https://cmake.org/files/v3.26/
- VS 2017 community = https://download.visualstudio.microsoft.com/download/pr/e84651e1-d13a-4bd2-a658-f47a1011ffd1/e17f0d85d70dc9f1e437a78a90dcfc527befe3dc11644e02435bdfe8fd51da27/vs_Community.exe

#### Erro no numpy
Pode ocorrer um erro ao tentar compilar o projeto devido à uma alteração no Numpy, abordada em https://github.com/davisking/dlib/issues/2463.

Para ajustar essa situação, é necessário encontrar o arquivo "pybind11/numpy.h".
E substituir o seguinte trecho de código:
```
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif
```
Para:
```
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
```

#### Resultados da compilação
Após compilar a API do OpenPose para Python, será necessário criar na pasta raiz dessa aplicação uma pasta chamada "openpose_build", e copiar para dentro dela as pastas listadas na documentação oficial da API.
Os caminhos dessas pastas são referenciadas em pose_analyzer.py.

### Libs de Python
Além do Python 3.7, listado anteriormente, é necessário instalar várias libs.
Para isso, bata consultar as presentes no arquivo requirements.txt.

### Compilação da aplicação para um executável.
Essa aplicação pode ser compilada para um execútavel através do pyinstaller.
Um tutorial para isso pode ser visto na documentação oficial do CustomTkinter: https://customtkinter.tomschimansky.com/documentation/packaging/.
Mas, para esse caso, será necessário adicionar alguns argumentos no comando de build, de forma que fique similar a:
```
pyinstaller --noconfirm --onedir --windowed `
--name "Treine Certo" `
--add-data "<CustomTkinter Location>/customtkinter;customtkinter/" `
--add-data "openpose_build/python/openpose/Release;openpose_build/python/openpose/Release" `
--add-data "openpose_build/x64/Release;openpose_build/x64/Release" `
--add-data "openpose_build/bin;openpose_build/bin" `
--add-data "openpose_build/models;openpose_build/models" `
--add-data "models/model_20250403_135511.pth;models" `
--add-data "front;front" `
--icon="front/icon.ico" `
interface.py
```

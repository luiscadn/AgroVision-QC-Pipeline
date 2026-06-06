# Formulación Matemática y Rigor Científico 

Este documento detalla el sustento matemático y teórico de los algoritmos de visión artificial y aprendizaje automático implementados en el pipeline **AgroVision-QC-Pipeline**, satisfaciendo los requerimientos del criterio de rigor científico bajo la acreditación ABET.

---

## 1. Procesamiento y Segmentación Digital de Imágenes

### A. Suavizado Gaussiano ~Gaussian Blur
El suavizado gaussiano se utiliza como filtro de paso bajo para atenuar el ruido de alta frecuencia y suavizar detalles innecesarios de textura antes del proceso de segmentación. Matemáticamente, consiste en realizar la convolución espacial de la imagen bidimensional $I(x, y)$ con un kernel gaussiano $G_\sigma(x, y)$:

$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Donde:
- $(x, y)$ representan las coordenadas espaciales relativas respecto al centro del kernel.
- $\sigma$ es la desviación estándar de la distribución, la cual parametriza el ancho y la intensidad del suavizado de la campana de Gauss.

La operación de suavizado en forma discreta se expresa como:

$$I_{\text{filtrada}}(x, y) = I(x, y) * G_\sigma(x, y) = \sum_{u=-k}^{k} \sum_{v=-k}^{k} I(x-u, y-v) G_\sigma(u, v)$$

---

### B. Umbralización Óptima de Otsu ~Otsu's Thresholding
El criterio de Otsu calcula automáticamente el umbral óptimo $t^*$ para binarizar una imagen en escala de grises de $L$ niveles $[0, 1, \dots, L-1]$, maximizando la varianza inter-clase ($\sigma_B^2$) de los dos grupos resultantes (fondo $C_0$ y objeto $C_1$):

$$t^* = \arg\max_{0 \le t < L} \sigma_B^2(t)$$

La varianza inter-clase $\sigma_B^2(t)$ para un umbral dado $t$ se define como:

$$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2$$

Donde las probabilidades acumuladas ($\omega_i$) y las medias de clase ($\mu_i$) se computan a partir del histograma normalizado $p_i$ como:

$$\omega_0(t) = \sum_{i=0}^{t} p_i, \quad \omega_1(t) = \sum_{i=t+1}^{L-1} p_i = 1 - \omega_0(t)$$

$$\mu_0(t) = \sum_{i=0}^{t} \frac{i \cdot p_i}{\omega_0(t)}, \quad \mu_1(t) = \sum_{i=t+1}^{L-1} \frac{i \cdot p_i}{\omega_1(t)}$$

La maximización de la varianza inter-clase equivale matemáticamente a la minimización de la varianza intra-clase ($\sigma_W^2(t)$):

$$\sigma_W^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)$$

---

### C. Clausura Morfológica
La clausura morfológica se define formalmente como la dilatación ($\oplus$) de un conjunto de píxeles $A$ por un elemento estructurante $B$, seguida secuencialmente por una erosión ($\ominus$) utilizando el mismo elemento:

$$A \bullet B = (A \oplus B) \ominus B$$

Donde:
1. **Dilatación ($\oplus$):**
   $$A \oplus B = \{z \in \mathbb{Z}^2 \mid (B)_z \cap A \neq \emptyset\}$$
2. **Erosión ($\ominus$):**
   $$A \ominus B = \{z \in \mathbb{Z}^2 \mid (B)_z \subseteq A\}$$

La clausura morfológica elimina huecos oscuros y une contornos finos en el interior del objeto segmentado (la fruta) sin alterar significativamente su área o su forma exterior global.

---

## 2. Modelado y Optimización de Redes Neuronales 

### A. Función de Pérdida de Entropía Cruzada Categórica
Para la clasificación multiclase (buena, media y mala), la red se entrena minimizando la pérdida de entropía cruzada. Para un lote de $N$ imágenes y $C$ clases de salida, la función de costo se define como:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Donde:
- $y_{i,c}$ es un indicador binario ($0$ o $1$) que representa si la clase $c$ es la etiqueta real (ground truth) para la muestra $i$.
- $\hat{y}_{i,c}$ es la probabilidad predicha por la red para la muestra $i$ y la clase $c$, calculada a través de la función de activación **Softmax** aplicada en la capa de salida:

$$\hat{y}_{i,c} = \frac{\exp(z_{i,c})}{\sum_{j=1}^{C} \exp(z_{i,j})}$$

Donde $z_{i,c}$ es el vector de logits (salida lineal cruda de la última capa densa).

---

### B. Algoritmo de Optimización Adam ~Adaptive Moment Estimation
El optimizador Adam actualiza dinámicamente la tasa de aprendizaje utilizando estimaciones sesgadas de los momentos de primer ($m_t$) y segundo ($v_t$) orden de los gradientes de la pérdida $\mathcal{L}$ con respecto a los parámetros de la red $\theta$:

1. **Cálculo del gradiente en el paso de tiempo $t$:**
   $$g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$$

2. **Actualización del momento de primer orden (media exponencial):**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

3. **Actualización del momento de segundo orden (varianza no centrada exponencial):**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

4. **Corrección de sesgo para los momentos de primer y segundo orden:**
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

5. **Regla de actualización de parámetros $\theta_t$:**
   $$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Donde:
- $\alpha$ es la tasa de aprendizaje base (learning rate, por defecto $0.001$).
- $\beta_1$ y $\beta_2$ son los coeficientes de decaimiento exponencial (típicamente $0.9$ y $0.999$, respectivamente).
- $\epsilon$ es un término de regularización para evitar la división por cero (típicamente $10^{-8}$).

---

## 3. Formulación Matemática de Métricas de Evaluación

Para analizar cuantitativamente el desempeño de los clasificadores en clases desbalanceadas, se definen las siguientes métricas para cada clase $c$:

### A. Precisión ~Precision
Mide la fracción de predicciones positivas que fueron realmente correctas:

$$P_c = \frac{TP_c}{TP_c + FP_c}$$

### B. Sensibilidad ~Recall
Mide la fracción de muestras positivas reales que fueron identificadas correctamente por el modelo:

$$R_c = \frac{TP_c}{TP_c + FN_c}$$

### C. F1-Score
Media armónica de la precisión y la sensibilidad, que proporciona una métrica balanceada:

$$F1_c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$$

Donde $TP_c$, $FP_c$ y $FN_c$ representan Verdaderos Positivos, Falsos Positivos y Falsos Negativos para la clase $c$, respectivamente.

---

### D. Agrupamiento Macro ~Macro-Averaging
Dado el desbalance de clases detectado en el EDA, el agrupamiento macro se define como la media aritmética simple de las métricas obtenidas para cada clase individual. Esto penaliza severamente el mal desempeño en clases minoritarias:

$$\text{Precision}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} P_c$$

$$\text{Recall}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} R_c$$

$$\text{F1-Score}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

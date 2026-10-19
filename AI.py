import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import matplotlib.pyplot as plt


def load_csv(filepath):
    data = np.loadtxt(filepath, delimiter=',')

    labels = data[:,0]
    pixels = data[:, 1:]

    mask = (labels == 3)|(labels == 7)
    x = pixels[mask] / 255.0
    y_raw = labels[mask]

    y = (y_raw == 7).astype(int).reshape(-1,1)

    return x, y
def sigmoid(z):
    return 1 / (1 + np.exp( -np.clip(z, -25, 25)))

x_train,y_train = load_csv("mnist_train/mnist_train.csv")

x_test,y_test = load_csv("mnist_test/mnist_test.csv")

rs = RandomState(MT19937(SeedSequence(123456789))) 

w = rs.normal(0,0.01,(784,1))
b = 0.0

lr = 1
epochs = 1000

for epoch in range(epochs):
    z = np.dot(x_train, w) + b
    predictions = sigmoid(z)

    error = predictions - y_train
    dw = np.dot(x_train.T, error) / len(x_train)
    db = np.sum(error) / len(x_train)


    w -= lr*dw
    b -= lr*db

    if epoch % 200 == 0:
        loss = -np.mean(y_train * np.log(predictions + 1e-15) + (1 - y_train) * np.log(1 - predictions + 1e-15))
        print(f"Эпоха {epoch}/{epochs} -> Loss: {loss:.4f}")

def evaluate_accuracy(X, y, w, b):
    preds = sigmoid(np.dot(X, w) + b) >= 0.5
    return np.mean(preds == y)

test_acc = evaluate_accuracy(x_test, y_test, w, b)
print(f"\nИтоговая точность (Accuracy) на тест-выборке: {test_acc * 100:.2f}%")

plt.figure(figsize=(5, 5))
# Использование coolwarm наглядно разделяет положительные и отрицательные веса
plt.imshow(w.reshape(28, 28), cmap='coolwarm')
plt.title("Визуализация весов модели $w$")
plt.colorbar(label='Значение веса')
plt.axis('off')
plt.show()

def fgsm_attack(X, y, w, b, epsilon):
    # 1. Считаем предсказания (вероятности) для текущих X
    z = np.dot(X, w) + b
    a = sigmoid(z)
    
    # 2. Вычисляем градиент по входу x для каждого объекта: (a - y) * w
    # Из-за матричной формы делаем внешнее произведение векторов для каждой строки
    # (a - y) имеет форму (m, 1), w имеет форму (784,)
    grad_x = (a - y).reshape(-1, 1) * w.reshape(1, -1)
    
    # 3. Применяем формулу FGSM
    X_adv = X + epsilon * np.sign(grad_x)
    
    # 4. Ограничиваем значения рамками [0, 1]
    return np.clip(X_adv, 0, 1)

# Оцениваем зависимость Accuracy от Epsilon
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
accuracies = []

for eps in epsilons:
    X_adv = fgsm_attack(x_test, y_test, w, b, eps)
    acc = evaluate_accuracy(X_adv, y_test, w, b)
    accuracies.append(acc)
    print(f"Epsilon: {eps:<5} | Accuracy под атакой: {acc * 100:.2f}%")

# Строим график зависимости
plt.figure(figsize=(6, 4))
plt.plot(epsilons, accuracies, marker='o', color='red')
plt.title("Зависимость Accuracy от силы атаки ($\epsilon$)")
plt.xlabel("Epsilon ($\epsilon$)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Визуализация примеров атакованных изображений (например, при eps = 0.15)
eps_to_show = 0.15
X_adv_example = fgsm_attack(x_test, y_test, w, b, eps_to_show)
z_adv = np.dot(X_adv_example, w) + b
preds_adv = (sigmoid(z_adv) >= 0.5).astype(int)

plt.figure(figsize=(10, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(X_adv_example[i].reshape(28, 28), cmap='gray')
    true_digit = 7 if y_test[i] == 1 else 3
    pred_digit = 7 if preds_adv[i] == 1 else 3
    plt.title(f"True: {true_digit}\nPred: {pred_digit}")
    plt.axis('off')

plt.suptitle(f"Примеры изображений под атакой FGSM ($\epsilon$ = {eps_to_show})")
plt.show()

# ---------------------------------------------------------------------------
# Пункт 6. Сравнение возмущения FGSM с весами модели
# ---------------------------------------------------------------------------
# grad_x = (a - y) * w, поэтому sign(grad_x) = sign(a - y) * sign(w).
# Для истинного класса 7 (y=1) множитель (a - y) < 0, значит возмущение = -sign(w),
# для класса 3 (y=0) множитель (a - y) > 0, значит возмущение = +sign(w).
# То есть атака просто двигает каждый пиксель в сторону, где w уменьшает правильный логит.
eps_cmp = 0.15
X_adv_cmp = fgsm_attack(x_test, y_test, w, b, eps_cmp)
delta = X_adv_cmp - x_test                 # реально добавленное возмущение (после clip)
sign_w = np.sign(w.reshape(-1))

idx3 = np.where(y_test.ravel() == 0)[0][0]  # первый пример класса 3
idx7 = np.where(y_test.ravel() == 1)[0][0]  # первый пример класса 7

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
panels = [
    (w.reshape(28, 28), "Веса $w$"),
    (delta[idx3].reshape(28, 28), "Возмущение, истинный класс 3"),
    (delta[idx7].reshape(28, 28), "Возмущение, истинный класс 7"),
    (np.mean(delta, axis=0).reshape(28, 28), "Среднее возмущение по тесту"),
]
for ax, (img, title) in zip(axes, panels):
    im = ax.imshow(img, cmap='coolwarm')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(rf"Веса и возмущение FGSM ($\epsilon$ = {eps_cmp})")
plt.show()

# Количественно: корреляция возмущения со знаком весов (знак зависит от класса)
for name, idx in (("3", idx3), ("7", idx7)):
    corr = np.corrcoef(delta[idx], sign_w)[0, 1]
    print(f"Корреляция возмущения (класс {name}) с sign(w): {corr:+.3f}")

# Доля пикселей, где знак возмущения совпадает с ожидаемым sign(a-y)*sign(w).
# Пиксели, где clip обнулил возмущение (0 у фона и 1 у штриха), не учитываем.
sign_ay = np.where(y_test.ravel() == 1, -1.0, 1.0).reshape(-1, 1)
expected = sign_ay * sign_w.reshape(1, -1)
changed = delta != 0
match = np.mean(np.sign(delta[changed]) == expected[changed])
print(f"Доля изменённых пикселей: {np.mean(changed) * 100:.1f}%, "
      f"из них знак совпал с sign(a-y)*sign(w): {match * 100:.1f}%")

# Изменение логита: z(x+delta) - z(x) = w^T delta <= eps * ||w||_1
dz = (np.dot(X_adv_cmp, w) - np.dot(x_test, w)).ravel()
print(f"eps*||w||_1 = {eps_cmp * np.sum(np.abs(w)):.2f}, "
      f"средний |изменение логита| = {np.mean(np.abs(dz)):.2f}")

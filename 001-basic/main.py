# ML Model #1: Predict house prices based on house size
#
# ===================
# Actual observations
# ===================
#
# -------------------------
# Size (m^2) | Actual Price
# -------------------------
# 50 | 100
# 100 | 200
# 150 | 300
# 200 | 400
# -------------------------
# y = wx
# y is predicted price
# w is model parameter (weight)
# x is house size

def main():
    size = 100
    actual_price = 200
    weight = 1
    learning_rate = 0.00001
    epochs = 10

    for epoch in range(epochs):
        predicted_price = weight * size
        gradient = 2 * size * (predicted_price - actual_price)
        weight = weight - learning_rate * gradient

        loss = (predicted_price - actual_price) ** 2

        print(
            f"Epoch: {epoch + 1}, "
            f"Weight: {weight: .4f}, "
            f"Loss: {loss: .2f}"
        )

    print("Final weight:", weight)


if __name__ == '__main__':
    main()

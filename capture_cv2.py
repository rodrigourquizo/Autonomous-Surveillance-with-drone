import cv2

# Inicializa la capturadora de video (puedes probar diferentes valores en '0' si tienes varios dispositivos)
cap = cv2.VideoCapture()

# Verifica si la captura se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la capturadora de video")
    exit()

while True:
    # Lee un frame de la capturadora de video
    ret, frame = cap.read()

    # Verifica si el frame se leyó correctamente
    if not ret:
        print("Error al leer el frame")
        break

    # Muestra el frame
    cv2.imshow('Capturadora USB', frame)

    # Detiene el bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la capturadora de video y cierra la ventana
cap.release()
cv2.destroyAllWindows()

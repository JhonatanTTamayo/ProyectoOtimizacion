import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def bisection_method(func, interval_a, interval_b, tolerance=1e-6, max_iterations=100):
    if func(interval_a) * func(interval_b) >= 0:
        raise ValueError("La función no cambia de signo en el intervalo dado.")
    
    root, iterations = None, 0
    while (interval_b - interval_a) / 2 > tolerance and iterations < max_iterations:
        midpoint = (interval_a + interval_b) / 2
        if func(midpoint) == 0:
            root = midpoint
            break
        elif func(midpoint) * func(interval_a) < 0:
            interval_b = midpoint
        else:
            interval_a = midpoint
        root = (interval_a + interval_b) / 2
        iterations += 1
    return root, iterations

def false_position_method(func, interval_a, interval_b, tolerance=1e-6, max_iterations=100):
    if func(interval_a) * func(interval_b) >= 0:
        raise ValueError("La función no cambia de signo en el intervalo dado.")
    
    root, iterations = None, 0
    while abs(interval_b - interval_a) > tolerance and iterations < max_iterations:
        root = (interval_a * func(interval_b) - interval_b * func(interval_a)) / (func(interval_b) - func(interval_a))
        if func(root) == 0:
            break
        elif func(root) * func(interval_a) < 0:
            interval_b = root
        else:
            interval_a = root
        iterations += 1
    return root, iterations

def newton_raphson_method(func, derivative_func, initial_guess, tolerance=1e-6, max_iterations=100):
    root, iterations = None, 0
    x = initial_guess
    while abs(func(x)) > tolerance and iterations < max_iterations:
        x = x - func(x) / derivative_func(x)
        root = x
        iterations += 1
    return root, iterations

def secant_method(func, initial_guess_a, initial_guess_b, tolerance=1e-6, max_iterations=100):
    root, iterations = None, 0
    x0, x1 = initial_guess_a, initial_guess_b
    while abs(func(x1)) > tolerance and iterations < max_iterations:
        x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        x0, x1 = x1, x2
        root = x2
        iterations += 1
    return root, iterations

def plot_function(func, interval_a, interval_b):
    x = np.linspace(interval_a, interval_b, 1000)
    y = func(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Función')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label='Eje x')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--', label='Eje y')
    plt.title('Gráfico de la Función')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    return plt

def least_squares_regression(x, y, degree=1):
    coefficients = np.polyfit(x, y, degree)
    return np.poly1d(coefficients)

def lagrange_interpolation(x, y, target_x):
    result = 0
    n = len(x)
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term = term * (target_x - x[j]) / (x[i] - x[j])
        result += term
    return result

def euler_method(func, initial_condition, step_size, num_steps):
    t_values = np.arange(0, num_steps * step_size, step_size)
    y_values = [initial_condition]
    for i in range(1, num_steps):
        y_next = y_values[-1] + step_size * func(t_values[i - 1], y_values[-1])
        y_values.append(y_next)
    return t_values, y_values

def trapezoidal_rule(func, interval_a, interval_b, num_intervals):
    h = (interval_b - interval_a) / num_intervals
    result = 0.5 * (func(interval_a) + func(interval_b))
    for i in range(1, num_intervals):
        result += func(interval_a + i * h)
    result *= h
    return result

def multiple_trapezoidal_rule(func, interval_a, interval_b, num_segments):
    h = (interval_b - interval_a) / num_segments
    result = 0.5 * (func(interval_a) + func(interval_b))
    result += sum(func(interval_a + i * h) for i in range(1, num_segments))
    result *= h
    return result

def simpson_rule(func, interval_a, interval_b, num_intervals):
    h = (interval_b - interval_a) / num_intervals
    result = func(interval_a) + func(interval_b)
    result += 4 * sum(func(interval_a + i * h) for i in range(1, num_intervals, 2))
    result += 2 * sum(func(interval_a + i * h) for i in range(2, num_intervals - 1, 2))
    result *= h / 3
    return result

def quadratic_interpolation(func, interval_a, interval_b):
    x_points = np.linspace(interval_a, interval_b, 100)
    y_points = func(x_points)

    # Encuentra el mínimo de la parábola utilizando scipy.optimize.minimize
    result = minimize(func, x0=(interval_a + interval_b) / 2, bounds=[(interval_a, interval_b)])

    optimal_x = result.x[0]
    optimal_y = func(optimal_x)

    # Grafica la parábola y el punto óptimo
    plt.figure(figsize=(8, 6))
    plt.plot(x_points, y_points, label='Función cuadrática')
    plt.scatter(optimal_x, optimal_y, color='red', label='Óptimo aproximado')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label='Eje x')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--', label='Eje y')
    plt.title('Interpolación Cuadrática')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    return plt, optimal_x

def gradient_descent(func, gradient, initial_point, learning_rate=0.1, tolerance=1e-6, max_iterations=100):
    point = np.array(initial_point)
    iterations = 0
    while np.linalg.norm(gradient(point)) > tolerance and iterations < max_iterations:
        point = point - learning_rate * gradient(point)
        iterations += 1
    return point, iterations

def plot_contour(func, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    plt.figure(figsize=(8, 6))
    contours = plt.contour(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(contours, label='Nivel de la función')
    plt.title('Función de Contorno')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.scatter(0, 0, color='red', marker='*', label='Mínimo global en (0, 0)')
    plt.legend()
    return plt

# Estilos CSS para mejorar la apariencia
custom_styles = """
    <style>
        body {
            color: #2a2a2a;
            background-color: #f7f7f7;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #264653;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #e76f51;
        }
        .stButton {
            color: #fff !important;
            background-color: #e76f51 !important;
        }
        .stButton:hover {
            background-color: #2a9d8f !important;
        }
    </style>
"""


def main():
    st.set_page_config(layout="wide")
    st.title("Análisis de Datos con Streamlit")
    st.header("Bienvenido a nuestra plataforma de análisis")

    st.markdown("<h2 style='color: #e76f51;'>Descubre y Optimiza</h2>", unsafe_allow_html=True)
    #st.image("https://i.imgur.com/XWcvyRD.png", use_column_width=True)

    st.markdown("<hr style='border-color: #e76f51;'>", unsafe_allow_html=True)

    # Barra lateral mejorada
    st.sidebar.markdown("<h2 style='color: #e76f51;'>Equipo de Desarrollo</h2>", unsafe_allow_html=True)

    integrante1 = {
        "nombre": "James Alejandro Aristizabal Bedoya",
        "correo": "jamesa.aristizabalb@autonoma.edu.co",
        "carrera": "Ingeniería Electrónica"
    }

    integrante2 = {
        "nombre": "Jhonatan Tamayo Suarez",
        "correo": "jhonatan.tamayos@autonoma.edu.co",
        "carrera": "Ingeniería de Sistemas"
    }

    # Cuadros de información con colores y animaciones
    st.sidebar.markdown(f"<div class='animated fadeIn' style='color: #2a9d8f; margin-bottom: 10px;'><strong>{integrante1['nombre']}</strong><br>Correo: {integrante1['correo']}<br>Carrera: {integrante1['carrera']}</div>", unsafe_allow_html=True)

    st.sidebar.markdown(f"<div class='animated fadeIn' style='color: #2a9d8f; margin-bottom: 20px;'><strong>{integrante2['nombre']}</strong><br>Correo: {integrante2['correo']}<br>Carrera: {integrante2['carrera']}</div>", unsafe_allow_html=True)

    st.sidebar.markdown("<h2 style='color: #e76f51;'>Enlaces</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='https://docs.google.com/document/d/1ZbeROGaORl4n-KD6x85P8fvSMlukZfa_knu7bFFx_O8/edit?usp=drive_link' target='_blank' class='animated fadeIn stButton'>Manual de Usuario</a>", unsafe_allow_html=True)

    selected_method = st.selectbox("Selecciona un Método", ["Bisección", "Falsa Posición", "Newton-Raphson", "Secante", 
    "Regla Trapezoidal", "Regla Trapezoidal Múltiple", "Regla de Simpson 1/3", "Método de Euler", "Regla Trapezoidal", 
    "Regla Trapezoidal Múltiple", "Regla de Simpson 1/3", "Método de Euler Mejorado", "Método del Gradiente",
    "Métodos Cerrados de Optimización", "Interpolación Cuadrática"])
    st.subheader(f"Método: {selected_method}")


    if selected_method == "Bisección":
        st.write("Encuentra la raíz de la función utilizando el Método de Bisección.")
        example_function_bisection = lambda x: x**3 - 6*x**2 + 11*x - 6
        example_interval_a_bisection = st.slider("Intervalo a", min_value=-10.0, max_value=10.0, value=-1.0)
        example_interval_b_bisection = st.slider("Intervalo b", min_value=-10.0, max_value=10.0, value=1.0)
        tolerance_bisection = 1e-6
        root_bisection, iterations_bisection = bisection_method(example_function_bisection, example_interval_a_bisection, example_interval_b_bisection, tolerance_bisection)
        st.write(f"Raíz encontrada: {root_bisection}")
        st.write(f"Número de iteraciones: {iterations_bisection}")
        st.pyplot(plot_function(example_function_bisection, example_interval_a_bisection, example_interval_b_bisection))

    elif selected_method == "Falsa Posición":
        st.write("Encuentra la raíz de la función utilizando el Método de Falsa Posición.")
        example_function_false_position = lambda x: x**3 - 6*x**2 + 11*x - 6
        example_interval_a_false_position = st.slider("Intervalo a", min_value=-10.0, max_value=10.0, value=-1.0)
        example_interval_b_false_position = st.slider("Intervalo b", min_value=-10.0, max_value=10.0, value=1.0)
        tolerance_false_position = 1e-6
        root_false_position, iterations_false_position = false_position_method(example_function_false_position, example_interval_a_false_position, example_interval_b_false_position, tolerance_false_position)
        st.write(f"Raíz encontrada: {root_false_position}")
        st.write(f"Número de iteraciones: {iterations_false_position}")
        st.pyplot(plot_function(example_function_false_position, example_interval_a_false_position, example_interval_b_false_position))

    elif selected_method == "Newton-Raphson":
        st.write("Encuentra la raíz de la función utilizando el Método de Newton-Raphson.")
        example_function_newton = lambda x: x**3 - 6*x**2 + 11*x - 6
        example_derivative_newton = lambda x: 3*x**2 - 12*x + 11
        example_initial_guess_newton = st.slider("Guess inicial", min_value=-10.0, max_value=10.0, value=-1.0)
        tolerance_newton = 1e-6
        root_newton, iterations_newton = newton_raphson_method(example_function_newton, example_derivative_newton, example_initial_guess_newton, tolerance_newton)
        st.write(f"Raíz encontrada: {root_newton}")
        st.write(f"Número de iteraciones: {iterations_newton}")
        st.pyplot(plot_function(example_function_newton, example_interval_a_bisection, example_interval_b_bisection))

    elif selected_method == "Secante":
        st.write("Encuentra la raíz de la función utilizando el Método de la Secante.")
        example_function_secant = lambda x: x**3 - 6*x**2 + 11*x - 6
        example_initial_guess_a_secant = st.slider("Guess inicial a", min_value=-10.0, max_value=10.0, value=-1.0)
        example_initial_guess_b_secant = st.slider("Guess inicial b", min_value=-10.0, max_value=10.0, value=1.0)
        tolerance_secant = 1e-6
        root_secant, iterations_secant = secant_method(example_function_secant, example_initial_guess_a_secant, example_initial_guess_b_secant, tolerance_secant)
        st.write(f"Raíz encontrada: {root_secant}")
        st.write(f"Número de iteraciones: {iterations_secant}")
        st.pyplot(plot_function(example_function_secant, example_initial_guess_a_secant, example_initial_guess_b_secant))

    elif selected_method == "Método de Euler":
        st.write("Realiza integración numérica utilizando el Método de Euler.")
        example_function_euler = lambda t, y: -y
        initial_condition_euler = st.slider("Condición inicial", min_value=0.1, max_value=2.0, value=1.0)
        step_size_euler = st.slider("Tamaño de paso", min_value=0.01, max_value=0.5, value=0.1)
        num_steps_euler = 100
        result_euler = euler_method(example_function_euler, initial_condition_euler, step_size_euler, num_steps_euler)
        st.line_chart({'x': result_euler[0], 'y': result_euler[1]})

    elif selected_method == "Regresión por Mínimos Cuadrados":
        st.write("Ajusta una curva a tus datos utilizando el Método de Regresión por Mínimos Cuadrados.")
        st.write("Ingresa los datos:")
        data_size_regression = st.slider("Número de puntos de datos", min_value=2, max_value=10, value=5)
        input_data_regression = st.text_area("Ingresa los puntos de datos (x, y) uno por línea.", value="0, 2\n1, 3\n2, 1\n3, 5\n4, 4")
        data_regression = [list(map(float, line.split(","))) for line in input_data_regression.split("\n") if line]
        x_regression, y_regression = zip(*data_regression)
        degree_regression = st.slider("Grado del polinomio", min_value=1, max_value=10, value=2)
        regression_model = least_squares_regression(x_regression, y_regression, degree_regression)

        # Plot original data
        plt.scatter(x_regression, y_regression, label='Datos originales', color='blue')

        # Plot regression curve
        x_regression_fit = np.linspace(min(x_regression), max(x_regression), 100)
        y_regression_fit = regression_model(x_regression_fit)
        plt.plot(x_regression_fit, y_regression_fit, label=f'Regresión (grado {degree_regression})', color='red')

        plt.title('Regresión por Mínimos Cuadrados')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        st.pyplot(plt)

    elif selected_method == "Regla Trapezoidal":
        st.write("Realiza integración numérica utilizando la Regla Trapezoidal.")
        example_function_trapezoidal = lambda x: x**2
        example_interval_a_trapezoidal = st.slider("Intervalo a", min_value=0.0, max_value=2.0, value=0.0)
        example_interval_b_trapezoidal = st.slider("Intervalo b", min_value=0.0, max_value=2.0, value=2.0)
        num_intervals_trapezoidal = st.slider("Número de intervalos", min_value=1, max_value=10, value=1)
        result_trapezoidal = trapezoidal_rule(example_function_trapezoidal, example_interval_a_trapezoidal, example_interval_b_trapezoidal, num_intervals_trapezoidal)
        st.write(f"Resultado de la integración: {result_trapezoidal}")

    elif selected_method == "Regla Trapezoidal Múltiple":
        st.write("Realiza integración numérica utilizando la Regla Trapezoidal Múltiple.")
        example_function_multi_trapezoidal = lambda x: x**2
        example_interval_a_multi_trapezoidal = st.slider("Intervalo a", min_value=0.0, max_value=2.0, value=0.0)
        example_interval_b_multi_trapezoidal = st.slider("Intervalo b", min_value=0.0, max_value=2.0, value=2.0)
        num_segments_multi_trapezoidal = st.slider("Número de segmentos", min_value=1, max_value=10, value=1)
        result_multi_trapezoidal = multiple_trapezoidal_rule(example_function_multi_trapezoidal, example_interval_a_multi_trapezoidal, example_interval_b_multi_trapezoidal, num_segments_multi_trapezoidal)
        st.write(f"Resultado de la integración: {result_multi_trapezoidal}")

    elif selected_method == "Regla de Simpson 1/3":
        st.write("Realiza integración numérica utilizando la Regla de Simpson.")
        example_function_simpson = lambda x: x**2
        example_interval_a_simpson = st.slider("Intervalo a", min_value=0.0, max_value=2.0, value=0.0)
        example_interval_b_simpson = st.slider("Intervalo b", min_value=0.0, max_value=2.0, value=2.0)
        num_intervals_simpson = st.slider("Número de intervalos (debe ser par)", min_value=2, max_value=10, step=2, value=2)
        result_simpson = simpson_rule(example_function_simpson, example_interval_a_simpson, example_interval_b_simpson, num_intervals_simpson)
        st.write(f"Resultado de la integración: {result_simpson}")

    elif selected_method == "Interpolación de Lagrange":
        st.write("Interpola tus datos utilizando el Método de Interpolación de Lagrange.")
        st.write("Ingresa los datos:")
        data_size_interpolation = st.slider("Número de puntos de datos", min_value=2, max_value=10, value=5)
        input_data_interpolation = st.text_area("Ingresa los puntos de datos (x, y) uno por línea.", value="0, 2\n1, 3\n2, 1\n3, 5\n4, 4")
        data_interpolation = [list(map(float, line.split(","))) for line in input_data_interpolation.split("\n") if line]
        x_interpolation, y_interpolation = zip(*data_interpolation)
        target_x_interpolation = st.slider("Valor de x para interpolar", min_value=min(x_interpolation), max_value=max(x_interpolation), value=min(x_interpolation))
        interpolation_result = lagrange_interpolation(x_interpolation, y_interpolation, target_x_interpolation)

        # Plot original data
        plt.scatter(x_interpolation, y_interpolation, label='Datos originales', color='blue')

        # Plot interpolation result
        plt.scatter(target_x_interpolation, interpolation_result, label='Interpolación', color='red')

        plt.title('Interpolación de Lagrange')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        st.pyplot(plt)
    
    elif selected_method == "Interpolación Cuadrática":
        st.write("Aproxima un máximo o mínimo de la función a una parábola.")
        example_function_quadratic = lambda x: x**2
        example_interval_a_quadratic = st.slider("Intervalo a", min_value=-10.0, max_value=10.0, value=-1.0)
        example_interval_b_quadratic = st.slider("Intervalo b", min_value=-10.0, max_value=10.0, value=5.0)
        plot_quadratic, optimal_x_quadratic = quadratic_interpolation(example_function_quadratic, example_interval_a_quadratic, example_interval_b_quadratic)
        st.write(f"Óptimo aproximado: {optimal_x_quadratic}")
        st.pyplot(plot_quadratic)

    elif selected_method == "Método del Gradiente":
        st.write("Encuentra un mínimo local de la función utilizando el Método del Gradiente.")
        st.write("Ejemplo de función: $f(x, y) = x^2 + y^2$")

        example_function_gradient = lambda x: x[0]**2 + x[1]**2
        example_gradient = lambda x: np.array([2 * x[0], 2 * x[1]])

        plot_contour(example_function_gradient, [-5, 5], [-5, 5])
        st.pyplot()

        initial_point_gradient = st.text_input("Ingresa el punto inicial como una lista [x, y]", "[2, 2]")
        initial_point_gradient = eval(initial_point_gradient)

        result_gradient, iterations_gradient = gradient_descent(example_function_gradient, example_gradient, initial_point_gradient)
        st.write(f"Mínimo local encontrado en el punto: {result_gradient}")
        st.write(f"Número de iteraciones: {iterations_gradient}")


if __name__ == "__main__":
    main()
  




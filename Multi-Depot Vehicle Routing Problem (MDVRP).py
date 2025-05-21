import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def read_mdvrp_dat(filename):
    """
    Lee una instancia MDVRP desde un archivo .dat y devuelve la información en un diccionario.
    
    Formato asumido:
      - Primera línea: <param> <vehículos_por_deposito> <capacidad> <número_de_depositos>
      - Siguientes <número_de_depositos> líneas: coordenadas (x, y) de cada depósito
      - Resto de líneas: datos de clientes en formato:
             id  x  y  demanda  tiempo_servicio  [otros parámetros...]
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    header_vals = lines[0].split()
    param, vehicles_per_depot, capacity, num_depots = map(int, header_vals[:4])
    
    depots = []
    for i in range(1, 1 + num_depots):
        parts = lines[i].split()
        x, y = map(float, parts[:2])
        depots.append((x, y))
    
    customers = []
    for line in lines[1 + num_depots:]:
        parts = line.split()
        cust_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = int(parts[3])
        service = float(parts[4])
        extra = list(map(int, parts[5:])) if len(parts) > 5 else []
        customers.append({
            'id': cust_id,
            'x': x,
            'y': y,
            'demand': demand,
            'service': service,
            'extra': extra
        })
    
    return {
        'header': {
            'param': param,
            'vehicles_per_depot': vehicles_per_depot,
            'capacity': capacity,
            'num_depots': num_depots
        },
        'depots': depots,
        'customers': customers
    }

class BRKGA:
    def __init__(self, instance_data, pop_size=100, elite_percentage=0.2, 
                 mutant_percentage=0.1, bias_prob=0.7, generations=100):
        self.instance = instance_data
        self.pop_size = pop_size
        self.elite_size = int(pop_size * elite_percentage)
        self.mutant_size = int(pop_size * mutant_percentage)
        self.crossover_size = pop_size - self.elite_size - self.mutant_size
        self.bias_prob = bias_prob
        self.generations = generations
        self.chromosome_length = len(instance_data['customers']) * 2  # Una para asignación, otra para secuencia
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            chromosome = [random.random() for _ in range(self.chromosome_length)]
            self.population.append(chromosome)
    
    def decode_chromosome(self, chromosome):
        num_customers = len(self.instance['customers'])
        num_depots = self.instance['header']['num_depots']
        
        # Primera parte del cromosoma: asignación de clientes a depósitos
        depot_assignments = []
        for i in range(num_customers):
            depot_idx = int(chromosome[i] * num_depots)
            depot_assignments.append(depot_idx)
        
        # Segunda parte: prioridad para secuenciar clientes en rutas
        sequence_keys = chromosome[num_customers:]
        
        # Organizar clientes por depósito y prioridad
        depot_customers = {d: [] for d in range(num_depots)}
        for i, depot in enumerate(depot_assignments):
            customer = self.instance['customers'][i]
            priority = sequence_keys[i]
            depot_customers[depot].append((i, customer, priority))
        
        # Para cada depósito, crear rutas respetando capacidad
        routes = []
        for depot_idx, customers in depot_customers.items():
            # Ordenar clientes por prioridad
            customers.sort(key=lambda x: x[2])
            
            capacity = self.instance['header']['capacity']
            vehicles_available = self.instance['header']['vehicles_per_depot']
            vehicle_count = 0
            
            # Inicializar primera ruta
            current_route = []
            current_load = 0
            
            # Procesar todos los clientes asignados a este depósito
            for cust_idx, customer, _ in customers:
                # Comprobar si el cliente cabe en la ruta actual
                if current_load + customer['demand'] <= capacity:
                    # El cliente cabe en esta ruta, añadirlo
                    current_route.append(cust_idx)
                    current_load += customer['demand']
                else:
                    # El cliente no cabe, cerrar ruta actual y abrir nueva si hay vehículos disponibles
                    if current_route:  # Solo guardar si hay clientes en la ruta
                        routes.append({
                            'depot': depot_idx,
                            'customers': current_route.copy(),
                            'load': current_load
                        })
                        vehicle_count += 1
                    
                    # Comprobar si quedan vehículos disponibles
                    if vehicle_count >= vehicles_available:
                        # No hay más vehículos, ignorar cliente
                        continue
                    
                    # Iniciar nueva ruta con este cliente
                    current_route = [cust_idx]
                    current_load = customer['demand']
            
            # No olvidar la última ruta si contiene clientes
            if current_route and vehicle_count < vehicles_available:
                routes.append({
                    'depot': depot_idx,
                    'customers': current_route,
                    'load': current_load
                })
        
        return routes
    
    def calculate_fitness(self, routes):
        total_distance = 0
        
        for route in routes:
            depot_idx = route['depot']
            depot_x, depot_y = self.instance['depots'][depot_idx]
            
            # Distancia desde depósito al primer cliente
            if route['customers']:
                first_cust_idx = route['customers'][0]
                first_cust = self.instance['customers'][first_cust_idx]
                total_distance += self.distance(depot_x, depot_y, first_cust['x'], first_cust['y'])
                
                # Distancia entre clientes consecutivos
                prev_cust = first_cust
                for i in range(1, len(route['customers'])):
                    cust_idx = route['customers'][i]
                    cust = self.instance['customers'][cust_idx]
                    total_distance += self.distance(prev_cust['x'], prev_cust['y'], cust['x'], cust['y'])
                    prev_cust = cust
                
                # Distancia del último cliente de vuelta al depósito
                last_cust = self.instance['customers'][route['customers'][-1]]
                total_distance += self.distance(last_cust['x'], last_cust['y'], depot_x, depot_y)
        
        # Penalizaciones por restricciones
        penalty = 0
        
        return total_distance + penalty

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def evolve(self):
        for gen in range(self.generations):
            # Decodificar y evaluar toda la población
            solutions = []
            for chromosome in self.population:
                solution = self.decode_chromosome(chromosome)
                fitness = self.calculate_fitness(solution)
                solutions.append((chromosome, fitness))
            
            # Ordenar por fitness
            solutions.sort(key=lambda x: x[1])
            
            # Guardar la mejor solución encontrada
            if solutions[0][1] < self.best_fitness:
                self.best_fitness = solutions[0][1]
                self.best_solution = self.decode_chromosome(solutions[0][0])
            
            # Crear nueva población
            new_population = []
            
            # Mantener la élite
            elite = [s[0] for s in solutions[:self.elite_size]]
            new_population.extend(elite)
            
            # Generar mutantes aleatorios
            for _ in range(self.mutant_size):
                mutant = [random.random() for _ in range(self.chromosome_length)]
                new_population.append(mutant)
            
            # Realizar cruces sesgados
            non_elite = [s[0] for s in solutions[self.elite_size:]]
            for _ in range(self.crossover_size):
                elite_parent = random.choice(elite)
                non_elite_parent = random.choice(non_elite)
                child = self.biased_crossover(elite_parent, non_elite_parent)
                new_population.append(child)
            
            self.population = new_population
            
            if (gen + 1) % 10 == 0:
                print(f"Generación {gen+1}/{self.generations} - Mejor fitness: {self.best_fitness}")
    
    def biased_crossover(self, elite_parent, non_elite_parent):
        child = []
        for i in range(self.chromosome_length):
            if random.random() < self.bias_prob:
                child.append(elite_parent[i])
            else:
                child.append(non_elite_parent[i])
        return child

def plot_mdvrp_solution(instance, solution, best_fitness):
    plt.figure(figsize=(12, 10))
    
    # Colores para diferentes depósitos
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # Dibujar depósitos (como cuadrados)
    for i, (x, y) in enumerate(instance['depots']):
        plt.scatter(x, y, c=colors[i % len(colors)], marker='s', s=100)
        plt.text(x, y, f'D{i}', fontsize=12)
    
    # Dibujar clientes (como círculos)
    for i, customer in enumerate(instance['customers']):
        plt.scatter(customer['x'], customer['y'], c='gray', alpha=0.6)
        plt.text(customer['x'], customer['y'], f'{i}', fontsize=8)
    
    # Dibujar rutas
    for i, route in enumerate(solution):
        depot_idx = route['depot']
        depot_x, depot_y = instance['depots'][depot_idx]
        color = colors[depot_idx % len(colors)]
        
        if not route['customers']:
            continue
            
        # Conectar depósito con primer cliente
        first_cust_idx = route['customers'][0]
        first_cust = instance['customers'][first_cust_idx]
        plt.plot([depot_x, first_cust['x']], [depot_y, first_cust['y']], c=color, alpha=0.7)
        
        # Conectar clientes consecutivos
        for j in range(len(route['customers'])-1):
            cust1_idx = route['customers'][j]
            cust2_idx = route['customers'][j+1]
            cust1 = instance['customers'][cust1_idx]
            cust2 = instance['customers'][cust2_idx]
            plt.plot([cust1['x'], cust2['x']], [cust1['y'], cust2['y']], c=color, alpha=0.7)
        
        # Conectar último cliente con depósito
        last_cust_idx = route['customers'][-1]
        last_cust = instance['customers'][last_cust_idx]
        plt.plot([last_cust['x'], depot_x], [last_cust['y'], depot_y], c=color, alpha=0.7)
    
    plt.title(f'MDVRP Solución - Distancia total: {best_fitness:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_depot_distribution(solution, instance_data):
    depot_count = instance_data['header']['num_depots']
    depot_customers = {i: 0 for i in range(depot_count)}
    depot_load = {i: 0 for i in range(depot_count)}
    
    for route in solution:
        depot_idx = route['depot']
        depot_customers[depot_idx] += len(route['customers'])
        depot_load[depot_idx] += route['load']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de clientes por depósito
    depots = list(depot_customers.keys())
    customers = list(depot_customers.values())
    ax1.bar(depots, customers, color='skyblue')
    ax1.set_xlabel('Depósito')
    ax1.set_ylabel('Número de Clientes')
    ax1.set_title('Distribución de Clientes por Depósito')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico de carga por depósito
    loads = list(depot_load.values())
    ax2.bar(depots, loads, color='lightgreen')
    ax2.set_xlabel('Depósito')
    ax2.set_ylabel('Carga Total')
    ax2.set_title('Distribución de Carga por Depósito')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_route_loads(solution, instance_data):
    capacity = instance_data['header']['capacity']
    route_loads = [route['load'] for route in solution]
    route_indices = list(range(1, len(route_loads) + 1))
    utilization = [load / capacity * 100 for load in route_loads]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico de carga absoluta
    bars = ax1.bar(route_indices, route_loads, color='cornflowerblue')
    ax1.axhline(y=capacity, color='red', linestyle='--', label=f'Capacidad ({capacity})')
    ax1.set_xlabel('Ruta')
    ax1.set_ylabel('Carga')
    ax1.set_title('Carga por Ruta')
    ax1.set_xticks(route_indices)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir etiquetas de depósito
    for i, bar in enumerate(bars):
        depot = solution[i]['depot']
        ax1.text(bar.get_x() + bar.get_width()/2, 5, f'D{depot}', 
                ha='center', va='bottom', color='white', fontweight='bold', rotation=90)
    
    # Gráfico de utilización de capacidad
    colors = ['green' if u <= 75 else 'orange' if u <= 90 else 'red' for u in utilization]
    ax2.bar(route_indices, utilization, color=colors)
    ax2.set_xlabel('Ruta')
    ax2.set_ylabel('Utilización (%)')
    ax2.set_title('Porcentaje de Utilización de Capacidad por Ruta')
    ax2.set_xticks(route_indices)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de utilización
    avg_util = sum(utilization) / len(utilization)
    print(f"Utilización promedio de capacidad: {avg_util:.2f}%")
    print(f"Rutas con utilización > 90%: {sum(1 for u in utilization if u > 90)}")
    print(f"Rutas con utilización < 50%: {sum(1 for u in utilization if u < 50)}")

def plot_convergence_with_bks(brkga_instance, instance_name, bks_value, save_results=True):
    """
    Visualiza la convergencia del algoritmo con referencia al BKS.
    """
    if not hasattr(brkga_instance, 'fitness_history'):
        print("Esta instancia de BRKGA no tiene registro de fitness_history.")
        return
    
    # Crear gráfico con escala logarítmica
    plt.figure(figsize=(12, 7))
    plt.semilogy(brkga_instance.fitness_history, 'b-')
    
    # Marcar BKS
    if bks_value:
        plt.axhline(y=bks_value, color='orange', linestyle='--', label=f'BKS: {bks_value:.2f}')
    
    # Marcar fitness inicial y final
    initial_fitness = brkga_instance.fitness_history[0]
    final_fitness = brkga_instance.fitness_history[-1]
    
    plt.plot(0, initial_fitness, 'ro', markersize=8, label='Fitness Inicial')
    plt.plot(len(brkga_instance.fitness_history)-1, final_fitness, 'go', markersize=8, label='Fitness Final')
    
    # Calcular gap final
    if bks_value:
        gap = (final_fitness - bks_value) / bks_value * 100
        gap_text = f"Gap: {gap:.2f}%"
        plt.text(0.85, 0.1, gap_text, transform=plt.gca().transAxes, bbox=dict(facecolor='wheat', alpha=0.5))
    
    # Calcular mejora
    improvement = (initial_fitness - final_fitness) / initial_fitness * 100
    improvement_text = f"Mejora: {improvement:.2f}%"
    plt.text(0.85, 0.05, improvement_text, transform=plt.gca().transAxes, bbox=dict(facecolor='lightblue', alpha=0.5))
    
    # Etiquetas
    plt.title(f'Convergencia del Algoritmo - Instancia {instance_name}', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness (Escala Logarítmica)', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Guardar o mostrar
    plt.tight_layout()
    if save_results:
        plt.savefig(f"graficos_mdvrp/{instance_name}_convergencia.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_improvement_phases(brkga_instance, instance_name, save_results=True):
    """
    Visualiza las fases de mejora durante la convergencia.
    """
    if not hasattr(brkga_instance, 'fitness_history'):
        print("Esta instancia de BRKGA no tiene registro de fitness_history.")
        return
    
    # Calcular mejora acumulada
    initial_fitness = brkga_instance.fitness_history[0]
    improvements = [(initial_fitness - f) / initial_fitness * 100 for f in brkga_instance.fitness_history]
    
    # Identificar fases de mejora
    total_gens = len(improvements)
    phase1_end = min(int(total_gens * 0.3), 200)  # 30% o max 200 generaciones
    phase2_end = min(int(total_gens * 0.7), 350)  # 70% o max 350 generaciones
    
    # Calcular mejora por fase
    phase1_improvement = improvements[phase1_end] if phase1_end < len(improvements) else 0
    phase2_improvement = improvements[phase2_end] - improvements[phase1_end] if phase2_end < len(improvements) else 0
    phase3_improvement = improvements[-1] - improvements[phase2_end] if phase2_end < len(improvements) else 0
    total_improvement = improvements[-1]
    
    # Crear gráfico
    plt.figure(figsize=(12, 7))
    
    # Graficar mejora acumulada
    plt.plot(improvements, 'b-', linewidth=2)
    
    # Colorear fases
    plt.axvspan(0, phase1_end, alpha=0.2, color='blue', label=f'Fase inicial: {phase1_improvement:.2f}%')
    plt.axvspan(phase1_end, phase2_end, alpha=0.2, color='orange', label=f'Fase intermedia: {phase2_improvement:.2f}%')
    plt.axvspan(phase2_end, total_gens, alpha=0.2, color='green', label=f'Fase final: {phase3_improvement:.2f}%')
    
    # Añadir texto con mejora total
    plt.text(0.85, 0.9, f"Mejora total: {total_improvement:.2f}%", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Etiquetas
    plt.title(f'Mejora Acumulada Durante la Convergencia - {instance_name}', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Mejora Acumulada (%)', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Guardar o mostrar
    plt.tight_layout()
    if save_results:
        plt.savefig(f"graficos_mdvrp/{instance_name}_fases_mejora.png", dpi=300)
        plt.close()
    else:
        plt.show()

def run_and_visualize_instance(filename, generations=100):
    instance_data = read_mdvrp_dat(filename)
    print(f"\nEjecutando instancia: {filename}")
    
    # Parámetros del algoritmo BRKGA
    brkga = BRKGA(
        instance_data=instance_data,
        pop_size=100,
        elite_percentage=0.2,
        mutant_percentage=0.15,
        bias_prob=0.7,
        generations=generations
    )
    
    brkga.initialize_population()
    brkga.evolve()
    
    print(f"Mejor fitness encontrado: {brkga.best_fitness:.2f}")
    plot_mdvrp_solution(instance_data, brkga.best_solution, brkga.best_fitness)
    plot_depot_distribution(brkga.best_solution, instance_data)
    plot_route_loads(brkga.best_solution, instance_data)
    
    return brkga

def run_all_instances(generations=100, save_results=True):
    """
    Procesa todas las instancias .dat en el directorio actual
    y devuelve los resultados agregados.
    """
    # Crear directorio para guardar gráficos si no existe
    if save_results and not os.path.exists("graficos_mdvrp"):
        os.makedirs("graficos_mdvrp")
    
    # Encontrar todos los archivos .dat
    dat_files = [f for f in os.listdir() if f.endswith('.dat')]
    
    # Preparar resultados
    results = []
    solutions = {}
    brkga_instances = {}
    
    # Procesar cada instancia
    for i, filename in enumerate(dat_files):
        print(f"\nProcesando instancia {i+1}/{len(dat_files)}: {filename}")
        instance_name = os.path.splitext(filename)[0]
        
        try:
            instance_data = read_mdvrp_dat(filename)
            
            # Crear y ejecutar BRKGA
            brkga = BRKGA(
                instance_data=instance_data,
                pop_size=100,
                elite_percentage=0.2,
                mutant_percentage=0.15,
                bias_prob=0.7,
                generations=generations
            )
            
            # Añadir atributo para seguir la historia del fitness
            brkga.fitness_history = []
            
            # Modificar método evolve para registrar el historial
            original_evolve = brkga.evolve
            def new_evolve():
                for gen in range(brkga.generations):
                    solutions = []
                    for chromosome in brkga.population:
                        solution = brkga.decode_chromosome(chromosome)
                        fitness = brkga.calculate_fitness(solution)
                        solutions.append((chromosome, fitness))
                    
                    # Ordenar por fitness
                    solutions.sort(key=lambda x: x[1])
                    
                    # Guardar la mejor solución encontrada
                    if solutions[0][1] < brkga.best_fitness:
                        brkga.best_fitness = solutions[0][1]
                        brkga.best_solution = brkga.decode_chromosome(solutions[0][0])
                    
                    # Registrar fitness actual
                    brkga.fitness_history.append(brkga.best_fitness)
                    
                    # El resto del código original
                    new_population = []
                    elite = [s[0] for s in solutions[:brkga.elite_size]]
                    new_population.extend(elite)
                    
                    for _ in range(brkga.mutant_size):
                        mutant = [random.random() for _ in range(brkga.chromosome_length)]
                        new_population.append(mutant)
                    
                    non_elite = [s[0] for s in solutions[brkga.elite_size:]]
                    for _ in range(brkga.crossover_size):
                        elite_parent = random.choice(elite)
                        non_elite_parent = random.choice(non_elite)
                        child = brkga.biased_crossover(elite_parent, non_elite_parent)
                        new_population.append(child)
                    
                    brkga.population = new_population
                    
                    if (gen + 1) % 10 == 0:
                        print(f"Generación {gen+1}/{brkga.generations} - Mejor fitness: {brkga.best_fitness:.2f}")
            
            # Reemplazar el método evolve
            brkga.evolve = new_evolve
            
            # Medir tiempo de ejecución
            import time
            start_time = time.time()
            
            # Inicializar y evolucionar la población
            brkga.initialize_population()
            brkga.evolve()
            
            # Calcular tiempo de ejecución
            execution_time = time.time() - start_time
            
            # Guardar resultados
            print(f"Mejor fitness encontrado: {brkga.best_fitness:.2f}")
            print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
            
            # Buscar BKS en el CSV de resultados si existe
            bks_value = None
            if os.path.exists("resultados_mdvrp.csv"):
                with open("resultados_mdvrp.csv", "r") as f:
                    for line in f:
                        if line.startswith(instance_name + ","):
                            parts = line.split(",")
                            if len(parts) > 2 and parts[2] and parts[2] != "":
                                bks_value = float(parts[2])
                                break
            
            # Visualizar y/o guardar resultados
            if not save_results:
                plot_mdvrp_solution(instance_data, brkga.best_solution, brkga.best_fitness)
                plot_depot_distribution(brkga.best_solution, instance_data)
                plot_route_loads(brkga.best_solution, instance_data)
            
            # Visualizar convergencia y fases de mejora
            plot_convergence_with_bks(brkga, instance_name, bks_value, save_results)
            plot_improvement_phases(brkga, instance_name, save_results)
            
            # Calcular estadísticas adicionales para resultados
            total_customers = 0
            total_load = 0
            for route in brkga.best_solution:
                total_customers += len(route['customers'])
                total_load += route['load']
            
            capacity = instance_data['header']['capacity']
            vehicles_per_depot = instance_data['header']['vehicles_per_depot']
            num_depots = instance_data['header']['num_depots']
            total_vehicles = num_depots * vehicles_per_depot
            
            # Calcular gap con BKS
            gap = None
            if bks_value:
                gap = (brkga.best_fitness - bks_value) / bks_value * 100
            
            # Añadir resultado
            result = {
                'instancia': instance_name,
                'fitness': brkga.best_fitness,
                'bks': bks_value,
                'gap': gap,
                'num_depots': num_depots,
                'num_customers': len(instance_data['customers']),
                'capacity': capacity,
                'vehicles_per_depot': vehicles_per_depot,
                'total_vehicles': total_vehicles,
                'tiempo': execution_time
            }
            results.append(result)
            solutions[instance_name] = brkga.best_solution
            brkga_instances[instance_name] = brkga
            
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
    
    # Guardar resultados en CSV
    if results:
        # Obtener todas las claves posibles
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        try:
            # Usar una ruta alternativa en la carpeta temporal del usuario
            import tempfile
            results_path = os.path.join(tempfile.gettempdir(), "resultados_mdvrp.csv")
            
            with open(results_path, "w") as f:
                # Escribir encabezado
                f.write(",".join(all_keys) + "\n")
                
                # Escribir filas
                for result in results:
                    # Formatear valores con 2 decimales para números flotantes
                    row = []
                    for key in all_keys:
                        value = result.get(key, "")
                        if isinstance(value, float):
                            row.append(f"{value:.2f}")  # Formatear con 2 decimales
                        else:
                            row.append(str(value))
                    f.write(",".join(row) + "\n")
            
            print(f"\nResultados guardados en {results_path}")
        except PermissionError:
            print("\n⚠️ ERROR: No se pudo guardar el archivo CSV de resultados.")
            print("   Posibles causas:")
            print("   - El archivo está abierto en otro programa (como Excel)")
            print("   - No tienes permisos de escritura en el directorio")
            print("   Soluciones:")
            print("   - Cierra cualquier programa que pueda estar usando el archivo")
            print("   - Ejecuta este script con permisos de administrador")
            print("   - Los resultados estarán disponibles en las variables devueltas por la función")
    
    # Guardar resultados en JSON
    if results:
        import json
        
        # Formatear los valores float a 2 decimales
        formatted_results = []
        for result in results:
            formatted_result = {}
            for key, value in result.items():
                if isinstance(value, float):
                    formatted_result[key] = round(value, 2)  # Redondear a 2 decimales
                else:
                    formatted_result[key] = value
            formatted_results.append(formatted_result)
        
        # Guardar en archivo JSON
        try:
            # Usar una ruta alternativa en la carpeta temporal del usuario
            import tempfile
            json_path = os.path.join(tempfile.gettempdir(), "resultados_mdvrp.json")
            
            with open(json_path, "w") as f:
                json.dump(formatted_results, f, indent=2)
            
            print(f"\nResultados guardados en {json_path}")
        except PermissionError:
            print("\n⚠️ ERROR: No se pudo guardar el archivo JSON de resultados.")
            print("   Los resultados estarán disponibles en las variables devueltas por la función")
    
    return results, solutions, brkga_instances

# Para ejecutar todas las instancias desde el programa principal
if __name__ == "__main__":
    choice = input("¿Deseas procesar (1) una instancia específica o (2) todas las instancias? [1/2]: ").strip()
    
    if choice == '1':
        # Código original para una instancia específica
        filename = input("Introduce el nombre del archivo .dat a usar (sin la extensión, si lo deseas): ").strip()
        if not os.path.exists(filename):
            filename_with_ext = filename + ".dat"
            if os.path.exists(filename_with_ext):
                filename = filename_with_ext
            else:
                print("El archivo no existe en el directorio actual.")
                exit(1)
        
        run_and_visualize_instance(filename)
    
    elif choice == '2':
        # Procesar todas las instancias
        save_option = input("¿Deseas guardar gráficos en archivos? [s/n]: ").strip().lower()
        save_results = save_option.startswith('s')
        
        generations = int(input("Número de generaciones a ejecutar (recomendado 50-200): ").strip())
        
        print("\nIniciando procesamiento de todas las instancias...")
        results, solutions, brkga_instances = run_all_instances(generations=generations, save_results=save_results)
    
    else:
        print("Opción no válida. Saliendo.")
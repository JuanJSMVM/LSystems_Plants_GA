import turtle as lia
import random
import io
from PIL import Image
def gen_cads(axiom, product_rules, n_items):
    new_axiom=axiom
    for _ in range(n_items):
        list_axiom=[]
        for car in new_axiom:
            aux=car
            if car in product_rules.keys():
                choose=random.choice(product_rules[car])
                list_axiom.extend(list(choose))
            else:    
                list_axiom.append(aux)
        new_axiom=''.join(list_axiom)          
    return list(new_axiom)
def move_Turtle_WithMemory(screen,turtle,cad, init_theta,theta, size, moves, pos_in, line_color, arrow_color="black"):
    turtle.color(line_color, arrow_color)
    turtle.penup()
    turtle.setpos(pos_in)
    turtle.pendown()
    turtle.setheading(init_theta)
    actual_dir=[]
    actual_heading=[]
    for car in cad:
        if car in moves.keys():
            name_func=moves[car].__name__
            if name_func in ['right','left']:
                moves[car](theta)
            else:
                moves[car](size)
        elif(car=='['):
            actual_dir.append(turtle.pos())
            actual_heading.append(turtle.heading())
            turtle.penup()
        elif(car==']'):
            turtle.setpos(actual_dir[-1])
            turtle.setheading(actual_heading[-1])
            
            turtle.pendown()
            actual_dir.pop(-1)
            actual_heading.pop(-1)
def create_env(model):
    best_ind=model.pop.best_ind
    best_individuals=model.best_fitness_records
    prod_best_ind=[]





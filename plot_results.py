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
            if len(actual_dir) > 0 and len(actual_heading) > 0:
                turtle.setpos(actual_dir[-1])
                turtle.setheading(actual_heading[-1])
            
                turtle.pendown()
                actual_dir.pop(-1)
                actual_heading.pop(-1)
def save_fig(fig_title,cad, init_theta,theta, size, 
             pos_in, line_color, arrow_color="black"):
    turtle=lia.Turtle()
    screen=lia.Screen()
    turtle.hideturtle()
    moves={'F':turtle.forward,'G':turtle.forward,'+':turtle.right,'-':turtle.left}                
    screen.setup(width=1.0,height=1.0)
    screen.bgcolor('black')
    turtle._tracer(0, 0)
    move_Turtle_WithMemory(screen,turtle,cad, init_theta,theta, 
                           size, moves, pos_in, line_color, arrow_color="black")
    turtle._update()
    
    ps = screen.getcanvas().postscript(colormode='color')
    lia.bye()
    # Crear una imagen PIL desde la representación PostScript
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img.save(fig_title+".png", "png")
def create_env(best_ind,best_records):
    
    best_ind=''.join(best_ind)
    best_individuals=best_records
    prod_best_ind={'F':[best_ind]}
    
    tree_best=gen_cads('F',prod_best_ind, 4)
    
    a=0
    save_fig("best_individual",tree_best,90,25,15,(0,-100),"#f4511e")
    for num,rule in enumerate(best_individuals):
        ind=''.join(rule)
        prod_ind={'F':[ind]}
        tree=gen_cads('F',prod_ind, 4)
        save_fig("ind_"+str(num),tree,90,25,15,(0,-100),"#f4511e")
        a+=1
        if a==10:
            return


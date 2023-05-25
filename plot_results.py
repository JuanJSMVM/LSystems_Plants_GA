import turtle as lia
import random
import io
from PIL import Image
import os
import imageio
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
    lia.TurtleScreen._RUNNING=True
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
            #turtle.penup()
            turtle.pendown()
        elif(car==']'):
            
            turtle.setpos(actual_dir[-1])
            turtle.setheading(actual_heading[-1])
        
            #turtle.pendown()
            turtle.penup()
            actual_dir.pop(-1)
            actual_heading.pop(-1)
def save_fig(fig_title,cad, init_theta,theta, size, 
             pos_in, line_color, arrow_color="black"):
    
    screen=lia.Screen()
    lia.TurtleScreen._RUNNING=True
    turtle=lia.Turtle()
    #lia.hideturtle()
    
    moves={'F':turtle.forward,'G':turtle.forward,'+':turtle.right,'-':turtle.left}                
    screen.setup(width=1.0,height=1.0)
    screen.bgcolor('black')
    turtle._tracer(0, 0)
    move_Turtle_WithMemory(screen,turtle,cad, init_theta,theta, 
                        size, moves, pos_in, line_color, arrow_color)
    turtle._update()
    
    ps = screen.getcanvas().postscript(colormode='color')
    #lia.done()
    #screen.mainloop()
    screen.bye()
    # Crear una imagen PIL desde la representación PostScript
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img.save(fig_title)
        

    
def create_env(best_ind,best_records,main_folder):
    
    best_ind=''.join(best_ind)
    best_individuals=best_records
    prod_best_ind={'F':[best_ind]}
    
    tree_best=gen_cads('F',prod_best_ind, 4)
    
    
    root=os.path.join(os.getcwd(),main_folder)
    if not os.path.exists(root):
        os.mkdir(root)
    path_img_best=os.path.join(root,"best_individual.png")
    save_fig(path_img_best,tree_best,90,29,15,(0,-400),"#f4511e")
    
    for num,rule in enumerate(best_individuals):
        ind=''.join(rule)
        prod_ind={'F':[ind]}
        tree=gen_cads('F',prod_ind, 4)
        path_img_=os.path.join(root,"ind_"+str(num)+".png")
        #print(prod_ind)
        save_fig(path_img_,tree,90,29,15,(0,-400),"#f4511e")
        #time.sleep(2)

def create_gif(main_folder):
    root=os.path.join(os.getcwd(),main_folder)
    img_arr = []
    name_img="ind_"
    ext=".png"
    print(root)
    for ind_num in range(70):
        name=name_img+str(ind_num)+ext
        path_img=os.path.join(root, name)
        img=Image.open(path_img)
        img_arr.append(img)
    path_gif=os.path.join(root, "best_gens_inds.gif")
    imageio.mimsave(path_gif, img_arr)
    
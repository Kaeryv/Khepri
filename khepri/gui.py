from raylib import *
from types import SimpleNamespace
from copy import copy

sw = 800
sh = 450

SetConfigFlags(FLAG_WINDOW_RESIZABLE);
InitWindow(sw, sh, b"Bast GUI")
MaximizeWindow()
SetTargetFPS(60)

camera = ffi.new("struct Camera3D *", [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 45.0, 0])
from collections import namedtuple
Material = namedtuple("Material", "name epsilon")

materials = [
        SimpleNamespace(name="Si", epsilon=3.0, color=RED),
        SimpleNamespace(name="Air", epsilon=1.0, color=SKYBLUE)
    ]
mnames = [m.name for m in materials]
layers = []
a = 1

selected_layer = -1
active_textbox = "none"
hovering_textbox = False
MAX_INPUT_CHARS=16
text=None
def float_input_box(name, x, y, w, h, number):
    global hovering_textbox, active_textbox, text
    DrawRectangle(x+2, y, w-3, h, WHITE)
    
    if text is None:
        text = list(map(ord, list(str(number))))
    if name == active_textbox:
        DrawRectangleLines(x, y, w-3, h, BLUE)
        key = GetCharPressed()
        cursor = len(text)
        while key > 0:
            if (key >= 32) and (key <= 125) and (len(text) < MAX_INPUT_CHARS):
                if cursor == len(text):
                    text.append(key)
                text[cursor] = key

            key = GetCharPressed()  
        if IsKeyPressed(KEY_BACKSPACE):
            if len(text) > 0:
                del text[-1]
    else:
        DrawRectangleLines(x, y, w-3, h, BLACK)
        if CheckCollisionPointRec(GetMousePosition(), (x,y,w-3,h)):
            hovering_textbox |= True
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)):
                if name != active_textbox:
                    text = None
                active_textbox = copy(name)
                return number

    jtext = "".join(map(chr, text))
    DrawText(jtext.encode("utf-8"), x+2, y, h-2, BLACK)

    if IsKeyPressed(KEY_ENTER):
        active_textbox = "none"
        text = None
        return float(jtext) 
    else:
        return number

def categ_input_box(x, y, w, h, choices, current):
    DrawRectangle(x, y, w-3, h, WHITE)
    DrawRectangleLines(x, y, w-3, h, BLACK)
    DrawText(f"{choices[current]} - {current+1}/{len(choices)}".encode("utf-8"), x+2, y, h-2, BLACK)
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)):
        if CheckCollisionPointRec(GetMousePosition(), (x,y,w-3,h)):
            current = (current + 1) % len(choices)

    return current



def draw_inspector(sw, sh, layers):
    iw, ih = sw//5, sh
    DrawRectangle(0, 0, iw, ih, GRAY)
    cur = 5
    DrawText(b"Simulation", 3, cur, 20, WHITE)
    cur += 20

    DrawText(b"> Layers", 3, cur, 20, WHITE)
    cur += 20

    for i, layer in enumerate(layers):
        idt1 = 10
        if i == selected_layer:
            DrawRectangle(3, cur, iw, 20, BLUE)
            DrawText(f"+ {layer.name}".encode("utf-8"), 3, cur, 20, BLACK)
            cur += 20
            DrawText(b"Depth", idt1, cur, 20, BLACK)
            cur += 20
            layer.depth =float_input_box(f"{layer.name}.depth", idt1, cur, iw, 20, layer.depth)
            cur += 20
            DrawText(b"Medium", idt1, cur, 20, BLACK)
            cur += 20
            new_index = categ_input_box(idt1, cur, iw, 20, mnames, mnames.index(layer.medium))
            layer.medium = mnames[new_index]
            cur += 20
        else:
            DrawRectangle(3, cur, iw, 20, GRAY)
            DrawText(f"+ {layer.name}".encode("utf-8"), 3, cur, 20, WHITE)
            cur += 20

    DrawText(b"> Materials", 3, cur, 20, WHITE)
    cur += 20
    
    for i, material in enumerate(materials):
        DrawRectangle(3, cur, iw, 20, GRAY)
        DrawText(f"+ {material.name}".encode("utf-8"), 3, cur, 20, WHITE)
        cur += 20




while not WindowShouldClose():
    hovering_textbox = False
    UpdateCamera(camera, CAMERA_ORBITAL)
    BeginDrawing()
    ClearBackground(RAYWHITE)
    BeginMode3D(camera[0])
    DrawGrid(40, 0.05)

    if IsWindowResized():
        sw = GetScreenWidth()
        sh = GetScreenHeight()


    if IsKeyPressed(KEY_A):
        layers.append(SimpleNamespace(name=f"L{len(layers)}", depth=0.3, medium="Si", structures=[]))

    
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)):
        cur_depth = 0.0
        for i, layer in enumerate(layers):
            ray = GetMouseRay(GetMousePosition(), camera[0]);
            px = 0.0
            py = cur_depth + layer.depth / 2
            pz = 0.0
            sx = 1.0
            sy = layer.depth
            sz = 1.0
            collision = GetRayCollisionBox(ray, [[ px - sx/2, py - sy/2, pz - sz/2 ],
                                              [ px + sx/2, py + sy/2, pz + sz/2 ]])
            cur_depth += layer.depth
            if collision.hit:
                if selected_layer != i:
                    selected_layer = i
                    text = None


    # Draw all layers
    cur_depth = 0.0
    for i, layer in enumerate(layers):
        DrawCube([0,cur_depth+layer.depth/2,0], 1.0, layer.depth, 1.0, materials[mnames.index(layer.medium)].color)
        cur_depth += layer.depth

    cur_depth = 0.0
    for i, layer in enumerate(layers):
        DrawCubeWires([0,cur_depth+layer.depth/2,0], 1.0, layer.depth,1.0,BLACK)
        if i == selected_layer:
            DrawCubeWires([0,cur_depth+layer.depth/2,0], 1.05, 1.05*layer.depth,1.05, GREEN)
        cur_depth += layer.depth


    EndMode3D()
    # Draw inspector
    draw_inspector(sw, sh, layers)

    if hovering_textbox:
        SetMouseCursor(MOUSE_CURSOR_IBEAM)
    else:
        SetMouseCursor(MOUSE_CURSOR_DEFAULT)
    EndDrawing()
CloseWindow()

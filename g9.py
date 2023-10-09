from js import document, window
import copy
from pyodide.ffi import create_proxy
import minitorch
import math
import numpy as np

#  User functions.
BACKEND = minitorch.make_tensor_backend(minitorch.NumpyOps, is_numpy=True)


def tensor(d, requires_grad=False):
    return minitorch.tensor(d, backend=BACKEND, requires_grad=requires_grad)

                
def point(loc, **kwargs):
    return {"type": "point", "c": loc} | kwargs


def line(loc, **kwargs):
    return {"type": "line", "c": loc} | kwargs


nn = minitorch


# This file is directly adapted from g9.js


def uncmin(params, loss, affects, end_on_line_search=True, tol=1e-8, maxit=1000):
    "Unconstrained minimization"
    def dot(x, y):
        return x @ y

    def norm2(x):
        return np.sqrt(x @ x)

    def out(x, y):
        return x.reshape(x.shape[0], 1) * y.reshape(1, y.shape[0])

    tol = max(tol, 2e-16)
    x0 = np.array([p[i] for n, p in params for i in range(p.shape[0])])
    f0 = loss()
    f0.backward()

    def construct_grad():
        g0 = np.array(
            [
                0.0
                if p.grad is None
                or (
                    affects is not None
                    and (
                        n not in affects
                        or (
                            isinstance(affects[n], minitorch.Tensor)
                            and affects[n][i] == 0
                        )
                    )
                )
                else p.grad[i]
                for n, p in params
                for i in range(p.shape[0])
            ]
        )
        return g0

    g0 = construct_grad()
    msg = ""
    n = g0.shape[0]
    H1 = np.eye(n)
    for it in range(maxit):
        step = -(H1 @ g0)
        nstep = norm2(step)
        if nstep < tol:
            msg = "Newton step smaller than tol"
            break
        t = 1
        df0 = g0 @ step

        # line search
        x1 = x0
        while it < maxit and t * nstep >= tol:
            s = step * t
            x1 = x0 + s

            i = 0
            for _, p in params:
                v = [x1[i + j] for j in range(p.shape[0])]
                p.update(tensor(v))
                p.requires_grad_(False)
                i += len(v)
            f1 = loss()
            if not (f1 - f0)[0] >= 0.1 * t * df0: 
                break
            t *= 0.5
            it += 1

        if t * nstep < tol and end_on_line_search:
            msg = "Line search step size smaller than tol"
            break
        if it == maxit:
            msg = "maxit reached during line search"
            break
        i = 0
        for _, p in params:
            v = [x1[i + j] for j in range(p.shape[0])]
            p.update(tensor(v))
            i += len(v)

        f1 = loss()
        f1.backward()
        g1 = construct_grad()

        y = g1 - g0
        ys = y @ s
        Hy = H1 @ y

        H1 = (H1 + (out(s, s) * (ys + (dot(y, Hy))) / (ys * ys))) - (
            (out(Hy, s)) + (out(s, Hy))
        ) / ys
        x0 = x1
        f0 = f1
        g0 = g1
    return {
        "solution": x0,
        "f": f0,
        "gradient": g0,
        "invHessian": H1,
        "iterations": it,
        "message": msg,
    }


# Drag JS Functions

def setAttributes(el, d):
    for k, v in d.items():
        el.setAttributeNS(None, k, v)


draggingCount = 0


def addDragHandler(el, startDrag):
    startex, startey = None, None

    def onstart(e):
        global draggingCount
        draggingCount += 1

        e.stopPropagation()
        e.preventDefault()

        onDrag = startDrag(e)

        startex = e.clientX
        startey = e.clientY
        # print("adding move")
        def onmove(e):
            e.preventDefault()
            e = e.touches[0] if hasattr(e, "touches") else e
            # print("move", e.clientX, e.clientY)
            onDrag(e.clientX - startex, e.clientY - startey)

        onmove = create_proxy(onmove)

        def onend(e):
            global draggingCount
            e.preventDefault()

            draggingCount -= 1
            document.removeEventListener("mousemove", onmove)
            document.removeEventListener("touchmove", onmove)
            document.removeEventListener("touchend", onend)
            document.removeEventListener("touchcancel", onend)
            document.removeEventListener("mouseup", onend)

        onend = create_proxy(onend)

        document.addEventListener("touchmove", onmove)
        document.addEventListener("mousemove", onmove)
        document.addEventListener("touchend", onend)
        document.addEventListener("touchcancel", onend)
        document.addEventListener("mouseup", onend)

    # print("adding start")
    onstart = create_proxy(onstart)
    el.addEventListener("touchstart", onstart)
    el.addEventListener("mousedown", onstart)

# Internal Shape classes    

class Point:
    def __init__(self, affects=None, g9=None):
        self.affects = None

    def mount(self, id, container, minimize_args):
        self.container = container
        self.el = document.createElementNS("http://www.w3.org/2000/svg", "circle")
        setAttributes(self.el, {"id": id})
        self.container.appendChild(self.el)
        self.el.setAttributeNS(None, "r", 5)

        def dragstart(e):
            c = self.args["c"]
            c = tensor([c[0], c[1]])

            def ret(dx, dy):
                d = tensor([dx, dy])

                def lossfn(args):
                    c2 = args["c"]
                    d2 = c2 - (c + d)
                    return (d2 * d2).sum()

                minimize_args(id, lossfn, self.args.get("affects", None))

            return ret

        addDragHandler(self.el, dragstart)

    def unmount(self):
        self.container.removeChild(self.el)

    def update(self, args):
        self.args = args
        my_args = copy.copy(args)
        my_args["cx"] = args["c"][0]
        my_args["cy"] = args["c"][1]
        del my_args["c"]
        if "affects" in my_args:
            del my_args["affects"]
        del my_args["type"]
        setAttributes(self.el, my_args)


class Line:
    def __init__(self, affects=None, g9=None):
        self.affects = None
        self.g9 = g9

    def unmount(self):
        self.container.removeChild(self.el)

    def update(self, args):
        self.args = args
        my_args = copy.copy(args)
        my_args["x1"] = args["c"][0]
        my_args["y1"] = args["c"][1]
        my_args["x2"] = args["c"][2]
        my_args["y2"] = args["c"][3]
        if "affects" in my_args:
            del my_args["affects"]

        del my_args["c"]
        del my_args["type"]
        setAttributes(self.el, my_args)

    def mount(self, id, container, minimize_args):
        self.container = container
        self.el = document.createElementNS("http://www.w3.org/2000/svg", "line")
        setAttributes(self.el, {"id": id})
        self.container.appendChild(self.el)
        self.el.setAttributeNS(None, "stroke", "#000")

        def dragstart(e):
            c = self.args["c"]
            c = tensor([c[0], c[1], c[2], c[3]])

            cx = e.clientX - self.g9.g9Offset["left"]
            cy = e.clientY - self.g9.g9Offset["top"]

            dx1 = c[2] - c[0]
            dy1 = c[3] - c[1]
            dx2 = cx - c[0]
            dy2 = cy - c[1]
            r = math.sqrt(dx2 * dx2 + dy2 * dy2) / math.sqrt(dx1 * dx1 + dy1 * dy1)
            x1y1 = tensor([[1, 0]])
            xy2_xy1 = tensor([[-1, 1]])

            def ret(dx, dy):
                d = tensor([dx, dy])
                cxdx = tensor([[(cx + dx), (cy + dy)]])

                def lossfn(args):
                    c2 = args["c"]
                    c2 = c2.view(2, 2)
                    d = (x1y1 @ c2) + (xy2_xy1 @ c2) * r - cxdx
                    return (d * d).sum()

                minimize_args(id, lossfn, self.args.get("affects", None))

            return ret

        addDragHandler(self.el, dragstart)



# G9 manager

class G9:
    def __init__(self, module, onChange=lambda: {}):

        self.renderables = {}
        self.module = module
        self.onChange = onChange
        self.node = document.createElementNS("http://www.w3.org/2000/svg", "svg")
        self.parent = None
        self.xAlign = "center"
        self.yAlign = "center"
        self.width = 0
        self.height = 0
        self.top = 0
        self.left = 0
        self.xOffset = 0
        self.yOffset = 0
        self.elements = {}
        self.g9Offset = None

    def noRerender(self):
        return lambda _: self.resize(False)

    def resize(self, rerender=True):
        if self.parent is None:
            return
        r = self.parent.getBoundingClientRect()

        if self.xAlign == "left":
            self.xOffset = 0
        elif self.xAlign == "center":
            self.xOffset = r.width / 2
        else:
            self.xOffset = r.width

        if self.yAlign == "top":
            self.yOffset = 0
        elif self.yAlign == "center":
            self.yOffset = r.height / 2
        else:
            self.yOffset = r.height
        self.node.setAttribute(
            "viewBox",
            " ".join(map(str, [-self.xOffset, -self.yOffset, r.width, r.height])),
        )

        self.g9Offset = {"top": r.top + self.yOffset, "left": r.left + self.xOffset}
        if rerender:
            self.render()

    def insertInto(self, selector):
        if self.parent is not None:
            self.remove()

        if isinstance(selector, str):
            self.parent = document.querySelector(selector)
        else:
            self.parent = selector
        self.parent.innerHTML = ""
        self.parent.appendChild(self.node)

        window.addEventListener("load", create_proxy(lambda _: self.resize()))
        window.addEventListener("resize", create_proxy(lambda _: self.resize()))
        window.addEventListener("scroll", create_proxy(lambda _: self.resize()))

        self.resize()

    def align(self, xval="center", yval="center"):
        self.xAlign = xval
        self.yAlign = yval
        self.resize()
        return self

    def render(self):
        self.renderables = self.module()
        params = self.module.named_parameters()
        for id, renderable in self.renderables.items():
            if id not in self.elements:
                # print(id)
                self.elements[id] = (
                    Point(g9=self) if renderable["type"] != "line" else Line(g9=self)
                )

                def minimize(loss, affects):
                    uncmin(params, loss, affects)
                    self.render()

                def mini(id, f, affects=None):
                    def loss():
                        r = self.module(id)
                        # if loss is None:
                        #     loss = 0.0
                        return f(r[id])

                    return minimize(loss, affects)

                self.elements[id].mount(id, self.node, mini)

            self.elements[id].update(self.renderables[id])

        for id, element in self.elements.items():
            if id not in self.renderables:
                self.elements[id].unmount()
                del self.elements[id]




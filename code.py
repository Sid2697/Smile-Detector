import web
urls = (
    '/input', 'index'
)


class index:
    def GET(self):
        i = web.input(name=None)
        return render.index(i.name)


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
render = web.template.render('templates/')

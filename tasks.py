from incite import task, task_namespace


@task(namespace='this.that', name='the_other')
def say_hi(c):
    print("Hi!")


ns = task_namespace()
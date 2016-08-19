class Dummy(object):

    def __init__(self):
        self.star = [7, 8]

    name = property(lambda x: None)

    def increment_star(self):
        self.star += 1

    def return_star(self):
        return self.star



dumdum = Dummy()
print dumdum.name
# print "star: ", dumdum.star
# print "return_Star: ", dumdum.return_star()
# t = dumdum.return_star()
# print "t: ", t
# dumdum.increment_star()
# print "calling increment"
# print "t: ", t
# print "star: ", dumdum.star
# print "return_star: ", dumdum.return_star()


y = getattr(dumdum, 'star')

y += [1]

print "y: ", y
print "star: ", dumdum.star
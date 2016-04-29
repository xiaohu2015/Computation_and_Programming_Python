# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
import datetime
'''
第8章：抽象的数据类型与类
'''

class Person(object):
    '''
    创建一个人
    '''
    def __init__(self, name):
        self.name = name
        try:
            lastBlank = name.rindex(' ')
            self.lastname = name[lastBlank+1:]
        except:
            self.lastname = name
        self.birthday = None

    def getName(self):
        '''
        返回这个人的姓名
        '''
        return self.name
    def getLastname(self):
        '''
        返回这个人的姓
        '''
        return self.lastname
    def setBirthday(self, birthDate):
        '''
        设置生日
        '''
        self.birthday = birthDate
    def getAge(self):
        '''
        返回这个人当前年龄对应的天数
        '''
        if self.birthday is None:
            raise ValueError("Birthday has not been set.")
        #return (datetime.date.today() - self.birthday).days
        return datetime.date.today().year - self.birthday.year

    def __lt__(self, other):
        '''
        按姓名比较
        '''
        if self.lastname == other.lastname:
            return self.name < other.name
        return self.lastname < other.lastname

    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

class MITPerson(Person):
    '''
    继承父类Person
    '''
    #类变量
    _nextIDNum = 0
    def __init__(self, name):
        super().__init__(name)
        #Person.__init__(self, name)
        self.idNum = MITPerson._nextIDNum
        MITPerson._nextIDNum += 1

    def getIDNum(self):
        return self.idNum

    def __lt__(self, other):
        return self.idNum < other.idNum

    def isStudent(self):
        return isinstance(self, Student)

class Student(MITPerson):
    pass

class UG(Student):
    def __init__(self, name, classYear):
        super().__init__(name)
        self.year = classYear
    def getClass(self):
        return self.year

class Grad(Student):
    pass

class Grades(object):
    '''
    记录学生的成绩
    '''
    def __init__(self):
        self.students = []
        self.grades = {}
        self.isSorted = True

    def addStudent(self, student):
        if student in self.students:
            raise ValueError("Duplicate student.")
        self.students.append(student)
        self.grades[student.getIDNum()] = []
        self.isSorted = False

    def addGrade(self, student, grade):
        try:
            self.grades[student.getIDNum()].append(grade)
        except:
            raise ValueError("Student not in mapping")

    def getGrades(self, student):
        try:
            return self.grades[student.getIDNum()][:]
        except:
            raise ValueError("Student not in mapping.")

    def getStudents(self):
        if not self.isSorted:
            self.students.sort()
            self.isSorted = True
        return self.students[:]

def gardeReport(course):
    report = ''
    for s in course.getStudents():
        tot = 0
        numGrades = 0
        for g in course.getGrades(s):
            tot += g
            numGrades += 1
        try:
            average = tot/numGrades
            report += '\n' + str(s) + '\'s mean grade is ' + str(average)
        except ZeroDivisionError:
            report += '\n' + str(s) + " has no grades "
    return report




if __name__ == "__main__":
    ug1 = UG("Jane Doe", 2014)
    ug2 = UG("John Doe", 2015)
    ug3 = UG("David Henry", 2003)
    g1 = Grad("Billy Buckner")
    g2 = Grad('Bucky F. Dent')
    sixHundred = Grades()
    sixHundred.addStudent(ug1)
    sixHundred.addStudent(ug2)
    sixHundred.addStudent(g1)
    sixHundred.addStudent(g2)
    for s in sixHundred.getStudents():
        sixHundred.addGrade(s, 75)
    sixHundred.addGrade(g1, 25)
    sixHundred.addGrade(g2, 100)
    sixHundred.addStudent(ug3)
    print(gardeReport(sixHundred))

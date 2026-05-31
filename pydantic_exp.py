from pydantic import BaseModel

class Student(BaseModel):
    name : str
    age : int

student=Student(name="Steni",age="18")

print(student.name)
print(student.age)
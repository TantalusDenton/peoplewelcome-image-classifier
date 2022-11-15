from typing import List
from fastapi import FastAPI, HTTPException
from models import User, Gender, Role, UserUpdateRequest, Image
from uuid import UUID, uuid4
import flowerClassifier

#Start FastAPI
app = FastAPI()

db: List[User] = [
    User(
        id = UUID("4981100f-f377-440f-8985-64fa0f0ff5a2"),
        first_name = "Joe",
        last_name = "Zimmerman",
        gender = Gender.male,
        roles = [Role.admin, Role.user]
    ),

    User(
        id = uuid4(),
        first_name = "Agent",
        last_name = "Smith",
        gender = Gender.male,
        roles = [Role.user]
    )
]

images: List[Image] = [
    Image(
        image = flowerClassifier.pidgeon_path,
        label = flowerClassifier.label
    )
]

@app.get("/")
async def root():
    #await foo()
    return {"Hello": "Mundo"}

@app.get("/api/v1/users")
async def fetch_user():
    return db

@app.get("/api/v1/image-classifier")
async def fetch_label():
    return images

@app.post("/api/v1/users")
async def register_user(user: User):
    db.append(user)
    return {"id": user.id}


@app.delete("/api/v1/users/{user_id}")
async def delete_user(user_id: UUID):
    for user in db:
        if user.id == user_id:
            db.remove(user)
            return
    raise HTTPException(
        status_code = 404,
        detail = f"user with id: {user_id} does not exist"
    )

@app.put("/api/v1/users/{user_id}")
async def update_user(user_update: UserUpdateRequest, user_id: UUID):
    for user in db:
        if user.id == user_id:
            if user_update.first_name is not None:
                user.first_name = user_update.first_name
            return   
    raise HTTPException(
        status_code = 404,
        detail = f"user with id: {user_id} does not exist"
    )
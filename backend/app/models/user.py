from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from bson import ObjectId

from typing import Any
from pydantic_core import core_schema
from pydantic import GetJsonSchemaHandler

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(ObjectId),
                ]),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, Any]:
        return {"type": "string"}

class UserBase(BaseModel):
    username: str
    phone_number: str
    role: str  # "elderly" or "caretaker"
    expo_push_token: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    hashed_password: str

class UserResponse(UserBase):
    id: Optional[PyObjectId] = Field(alias="_id")

    class Config:
        simple_json = True
        json_encoders = {ObjectId: str}

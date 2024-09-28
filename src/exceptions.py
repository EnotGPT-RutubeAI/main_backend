from starlette.responses import JSONResponse


class SuccessResponse(JSONResponse):
    def __init__(self, data=None, status_code=200, status=True, headers=None, **kwargs):
        response_data = {"status": status}
        if data is not None:
            response_data.update(data)

        super().__init__(response_data, status_code, headers, **kwargs)
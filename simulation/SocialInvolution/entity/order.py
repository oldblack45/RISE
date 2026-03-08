from random import randint


class Order:
    def __init__(self, id,  id_sum, pickup_location, delivery_location, pickup_time, delivery_time, money, runner_step):
        self.id = id
        self.id_num = id_sum # 总编号
        self.pickup_location = pickup_location  # 取货地点，表示为一个坐标（例如，(0, 0)）。
        self.delivery_location = delivery_location  # 送货地点，表示为一个坐标（例如，(1, 1)）
        self.pickup_time = pickup_time  # 取货时间窗口，表示为一个时间范围（例如，(0, 10)），表示骑手可以在这个时间范围内到达取货地点
        self.delivery_time = delivery_time  # 送货时间窗口，表示为一个时间范围（例如，(0, 20)），表示骑手可以在这个时间范围内到达送货地点
        self.status = 'unprocessed'
        self.money = money
        self.delete_time = int(runner_step + 15)
        self.finish_time = 0
        self.platform_money = money * 0.1

        def merchant_info(self):
            return self.pickup_location 
  
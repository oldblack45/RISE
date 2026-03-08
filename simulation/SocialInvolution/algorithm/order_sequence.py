def distance(location, other):
    x1, y1 = location
    x2, y2 = other
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
def objective(route, orders):
    total_delay = 0
    total_distance = 0
    current_time = 0
    current_location = (0, 0)  # 初始位置，根据实际情况可能需要修改

    order_dict = {order.id: order for order in orders}
    for event_type, order_id in route:
        order = order_dict[order_id]
        travel_time = distance(current_location, order.pickup_location if event_type == 'pickup' else order.delivery_location)
        current_time += travel_time  # 增加从上一个位置到当前位置的行驶时间
        total_distance += travel_time  # 增加到总距离

        # 对于取货事件，如果骑手提前到达，则等待
        if event_type == 'pickup':
            current_location = order.pickup_location
            if current_time < order.pickup_time[0]:
                current_time = order.pickup_time[0]
        else:  # 对于送货事件，计算延迟
            current_location = order.delivery_location
            if current_time > order.delivery_time[1]:
                delay = current_time - order.delivery_time[1]
                total_delay += delay

    return total_delay + total_distance

def get_order_by_id(orders, id):
    for order in orders:
        if order.id == id:
            return order
def order_sequence_cal(orders, rider):
  # 初始化事件列表
    events = []
    for order in orders:
        events.append(('pickup', order.id))
        events.append(('delivery', order.id))
    # 对事件进行排序，优先处理取货，且送货需要确保取货已在路线中
    sorted_events = sorted(events, key=lambda x:get_order_by_id(orders,x[1]).pickup_time[0] if x[0] == 'pickup' else float('inf'))

    for event_type, order_id in sorted_events:
        order = get_order_by_id(orders, order_id)
        # 如果订单已处理或骑手订单数达到限制，则跳过
        if rider.order_count >= rider.max_orders:
            continue
        # 检查插入的位置是否合法，对于送货事件，还要检查取货事件是否已经在路线中
        valid_positions = []
        for i in range(len(rider.route) + 1):
            new_route = rider.route[:i] + [(event_type, order_id)] + rider.route[i:]
            # 对于送货事件，确保其对应的取货事件已经在路线中
            if event_type == 'delivery' and not any(et == 'pickup' and oid == order_id for et, oid in rider.route[:i]):
                continue
            valid_positions.append((i, objective(new_route, orders)))

        # 如果没有有效位置，则跳过此订单
        if not valid_positions:
            continue

        # 选择目标函数值最小的位置
        best_position, best_value = min(valid_positions, key=lambda x: x[1])

        # 更新骑手的路线，订单状态和骑手的订单计数
        rider.route = rider.route[:best_position] + [(event_type, order_id)] + rider.route[best_position:]
        print(f"Rider {rider.id}'s route after insertion: {rider.route}")
# People and Parcel share a ride
Problem description:

K taxis (located at point 0) are scheduled to serve transport requests including N passenger requests 1, 2, . . . , N and M parcel requests 1, 2, . . ., M. Passenger request i (i = 1, . . ., N)) has pickup point i and drop-off point i + N + M, and parcel request i (i = 1, . . . , M) has pickup point i + N and drop-off point i + 2N + M. d(i,j) is the travel distance from point i to point j (i, j = 0, 1, . . ., 2N + 2M). Each passenger must be served by a direct trip without interruption (no stopping point between the pickup point and the drop-off point of the passenger in each route). Each taxi k has capacity Q[k] for serving parcel requests. The parcel request i (i = 1, 2, . . ., M) has quantity q[i].
Compute the routes for taxis satifying above contraints such that the length of the longest route among K taxis is minimal (in order to balance between lengths of taxis).
A route of a taxi k is represented by a sequence of points visited by that route: r[0], r[1], . . ., r[Lk] in which r[0] = r[Lk] = 0 (the depot)

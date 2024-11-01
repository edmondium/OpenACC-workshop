module thrust
	interface
		subroutine reduce(data, n, sum) bind(C,name="thrust_reduce_wrapper")
			use iso_c_binding
			real(c_double) :: data(*)
			integer(c_int), value :: n
			real(c_double) :: sum
		end subroutine reduce
	end interface
end module thrust

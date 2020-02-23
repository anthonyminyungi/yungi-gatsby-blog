import React from 'react'
import { Link } from 'gatsby'
import Search from '../search'

import './index.scss'

const searchIndices = [
  { name: `Posts`, title: `Blog Posts`, hitComp: `PostHit` },
]

export const Header = ({ title, location, rootPath }) => {
  const isRoot = location.pathname === rootPath
  return (
    <>
      {isRoot && (
        <h1 className="home-header">
          <Link to={`/`} className="link">
            {title}
          </Link>
        </h1>
      )}
      <Search collapse indices={searchIndices} />
    </>
  )
}

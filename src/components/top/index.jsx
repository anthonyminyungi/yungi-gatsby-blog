import React from 'react'
import { Link } from 'gatsby'
import { GitHubIcon } from '../social-share/github-icon'
import Search from '../search'

import './index.scss'

const searchIndices = { name: `Posts`, title: `Blog Posts`, hitComp: `PostHit` }

export const Top = ({ title, location, rootPath }) => {
  const isRoot = location.pathname === rootPath
  return (
    <div className="top">
      {!isRoot && (
        <Link to={`/`} className="link">
          {title}
        </Link>
      )}
      {/* <Search collapse indices={searchIndices} /> */}

      {/* I don't know why, but this feature doesn't work as I thought.
      https://www.gatsbyjs.org/docs/adding-search-with-algolia/
      https://janosh.io/blog/gatsby-algolia-search */}
      <GitHubIcon />
    </div>
  )
}
